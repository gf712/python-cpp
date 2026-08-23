#include "Conversion/PythonToPythonBytecode/PythonToPythonBytecode.hpp"
#include "Conversion/PythonToPythonBytecode/PatternPopulators.hpp"
#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/Dialect.hpp"
#include "Dialect/Python/IR/PythonAttributes.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"
#include "ast/AST.hpp"
#include "executable/Mangler.hpp"
#include "executable/bytecode/instructions/BinaryOperation.hpp"
#include "executable/bytecode/instructions/GetAwaitable.hpp"
#include "executable/bytecode/instructions/Unary.hpp"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "utilities.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>

namespace mlir {
namespace py {

	namespace {
		// The ops whose regions py.br_yield terminates, and which a structural
		// pattern flattens into the enclosing CFG. A walk looking for the yields
		// that belong to *one* such op must stop at any other one, because the
		// nested op's own pattern owns everything inside it.
		//
		// This is BranchYieldOp's ParentOneOf list (PythonOps.td) — the sixth
		// region-bearing python op, py.class, is excluded because its region
		// becomes a separate function rather than being flattened in place.
		bool is_flattened_region_op(mlir::Operation *op)
		{
			static_assert(mlir::py::BranchYieldOp::hasTrait<
				mlir::OpTrait::HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerOp>::Impl>());
			return mlir::isa<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerOp>(op);
		}

		// True when `yield_op` is a loop-control (break/continue) yield that binds to
		// the loop *enclosing* `loop` rather than to `loop` itself — i.e. it sits in
		// `loop`'s orelse, which is not part of the loop body.
		bool binds_to_enclosing_loop(mlir::py::PyLoopOpInterface loop,
			mlir::py::BranchYieldOp yield_op)
		{
			return yield_op.getKind().has_value() && loop.isLoopOrelse(yield_op->getParentRegion());
		}

		// True when some loop nested in `region` still holds a break/continue that
		// binds to the loop being lowered — i.e. one sitting in that nested loop's
		// orelse. Such a yield cannot be rewritten yet: it lives in a region that has
		// not been flattened, so branching it to our target block would be a
		// cross-region block reference, which is invalid IR.
		//
		// The caller defers (fails the match) until the nested loop lowers and inlines
		// the yield into our region, the same innermost-first trick TryOpLowering uses
		// for nested trys. Terminates because the innermost such loop has nothing
		// nested to wait on.
		bool has_pending_nested_orelse_control(mlir::Region &region)
		{
			if (region.empty()) { return false; }
			bool pending = false;
			region.walk<WalkOrder::PreOrder>([&pending](mlir::Operation *op) {
				auto loop = mlir::dyn_cast<mlir::py::PyLoopOpInterface>(op);
				if (!loop) { return WalkResult::advance(); }
				loop.getLoopOrelseRegion().walk<WalkOrder::PreOrder>(
					[&pending, loop](mlir::py::BranchYieldOp yield_op) {
						if (binds_to_enclosing_loop(loop, yield_op)) { pending = true; }
					});
				// Only this loop's own orelse matters here; anything deeper is the
				// nested loop's problem and it defers on it in turn.
				return WalkResult::skip();
			});
			return pending;
		}

		// Shared walker used by both ForLoopOpLowering and WhileOpLowering to lower
		// the py.br_yield ops in a loop body to cf.br ops targeting the right block
		// (continue→condition / step, break→end).
		//
		// It stops at every nested flattened-region op: those own their own yields,
		// and a break/continue in a nested loop's orelse only becomes ours once that
		// loop has been flattened into our region (see
		// has_pending_nested_orelse_control, which is what makes that ordering hold).
		void replace_loop_branch_yields(mlir::PatternRewriter &rewriter,
			mlir::Region &region,
			mlir::Block *continue_target,
			mlir::Block *break_target)
		{
			region.walk<WalkOrder::PreOrder>(
				[&rewriter, continue_target, break_target](mlir::Operation *operation) {
					if (is_flattened_region_op(operation)) { return WalkResult::skip(); }
					auto yield_op = mlir::dyn_cast<mlir::py::BranchYieldOp>(operation);
					if (!yield_op) { return WalkResult::advance(); }
					rewriter.setInsertionPoint(yield_op);
					if (!yield_op.getKind().has_value()
						|| yield_op.getKind().value() == py::LoopOpKind::continue_) {
						rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield_op, continue_target);
					} else if (yield_op.getKind().value() == py::LoopOpKind::break_) {
						rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield_op, break_target);
					}
					return WalkResult::advance();
				});
		}

		// Rewrites a loop orelse region's normal-completion (kindless) py.br_yield ops
		// into branches to the loop's exit block.
		//
		// Only the kindless ones. A `break`/`continue` written in an orelse binds to
		// the loop *enclosing* this one, so those are left in place: once this region
		// is inlined they sit directly in the enclosing loop's body, where its own
		// replace_loop_branch_yields claims them.
		void replace_orelse_completion_yields(mlir::PatternRewriter &rewriter,
			mlir::Region &region,
			mlir::Block *exit_target)
		{
			if (region.empty()) { return; }
			region.walk<WalkOrder::PreOrder>([&rewriter, exit_target](mlir::Operation *operation) {
				if (is_flattened_region_op(operation)) { return WalkResult::skip(); }
				auto yield_op = mlir::dyn_cast<mlir::py::BranchYieldOp>(operation);
				if (!yield_op || yield_op.getKind().has_value()) { return WalkResult::advance(); }
				rewriter.setInsertionPoint(yield_op);
				rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield_op, exit_target);
				return WalkResult::advance();
			});
		}

		// Collects the loop-control (break/continue) kinds that appear directly
		// in `region` — i.e. that belong to the enclosing loop rather than to a
		// nested loop/try/with (whose own lowering owns them). Descends into
		// TryHandlerOp so except-handler bodies are scanned.
		void collect_loop_control_kinds(mlir::Region &region, llvm::SmallSet<int, 2> &kinds)
		{
			if (region.empty()) { return; }
			region.walk<WalkOrder::PreOrder>([&kinds](mlir::Operation *op) {
				if (mlir::isa<mlir::py::TryOp,
						mlir::py::ForLoopOp,
						mlir::py::WhileOp,
						mlir::py::WithOp>(op)) {
					return WalkResult::skip();
				}
				if (auto y = mlir::dyn_cast<mlir::py::BranchYieldOp>(op);
					y && y.getKind().has_value()) {
					kinds.insert(static_cast<int>(*y.getKind()));
				}
				return WalkResult::advance();
			});
		}

		// A `break`/`continue` leaving a try/with body is represented as a
		// `br_yield break_/continue_` marker that only the enclosing loop pass
		// can resolve (to the loop's exit / continue target). Because try/with
		// are flattened *before* the loops, we re-emit the marker here, after
		// the block's exception cleanup ops, so it survives region inlining and
		// ends up directly in the loop body region for the loop pass to lower.
		//
		// When the try has a `finally`, the marker cannot fire directly: the
		// finally must run first. `finally_exits` maps each loop-control kind to
		// the entry of a pre-built finally clone whose normal exit *is* the
		// marker, so we branch there instead (see build_finally_loop_exits).
		void forward_loop_control_yield(mlir::PatternRewriter &rewriter,
			const llvm::DenseMap<int, mlir::Block *> &finally_exits,
			mlir::py::BranchYieldOp yield_op)
		{
			if (finally_exits.empty()) {
				mlir::py::BranchYieldOp::create(
					rewriter, yield_op.getLoc(), yield_op.getKindAttr());
				return;
			}
			auto it = finally_exits.find(static_cast<int>(*yield_op.getKind()));
			ASSERT(it != finally_exits.end());
			mlir::cf::BranchOp::create(rewriter, yield_op.getLoc(), it->second);
		}

		// For each break/continue kind that escapes the try through its finally,
		// clone the (still-pristine) finally region onto a dedicated exit path
		// and rewrite the clone's normal-completion (kindless) terminators into
		// the loop-control marker. The result is a per-kind entry block: branch
		// to it to "run finally, then break/continue". Must be called before the
		// original finally region is rewritten/inlined below.
		llvm::DenseMap<int, mlir::Block *> build_finally_loop_exits(mlir::PatternRewriter &rewriter,
			mlir::py::TryOp op,
			mlir::Block *endBlock)
		{
			llvm::DenseMap<int, mlir::Block *> exits;
			if (op.getFinally().empty()) { return exits; }

			llvm::SmallSet<int, 2> kinds;
			collect_loop_control_kinds(op.getBody(), kinds);
			for (mlir::Region &handler : op.getHandlers()) {
				collect_loop_control_kinds(handler, kinds);
			}
			collect_loop_control_kinds(op.getOrelse(), kinds);

			for (int kind : kinds) {
				auto kind_attr = mlir::py::LoopOpKindAttr::get(
					rewriter.getContext(), static_cast<mlir::py::LoopOpKind>(kind));
				mlir::IRMapping mapping;
				rewriter.cloneRegionBefore(
					op.getFinally(), *endBlock->getParent(), endBlock->getIterator(), mapping);
				for (mlir::Block &orig : op.getFinally()) {
					auto *cloned = mapping.lookup(&orig);
					if (auto y = mlir::dyn_cast<mlir::py::BranchYieldOp>(cloned->getTerminator());
						y && !y.getKind().has_value()) {
						rewriter.setInsertionPoint(y);
						rewriter.replaceOpWithNewOp<mlir::py::BranchYieldOp>(y, kind_attr);
					}
				}
				exits[kind] = mapping.lookup(&op.getFinally().front());
			}
			return exits;
		}

		struct ForLoopOpLowering : public mlir::OpRewritePattern<mlir::py::ForLoopOp>
		{
			using OpRewritePattern<mlir::py::ForLoopOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::ForLoopOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				// Lower innermost-first: a break/continue in a nested loop's orelse
				// binds to us, but only becomes rewritable once that loop has been
				// flattened into our region.
				if (has_pending_nested_orelse_control(op.getBody())) { return failure(); }

				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto iterable = op.getIterable();
				rewriter.setInsertionPointToEnd(initBlock);
				auto iterator = mlir::emitpybytecode::GetIter::create(
					rewriter, op.getStep().getLoc(), iterable.getType(), iterable);

				// advance iterator
				auto iterator_next_block = rewriter.createBlock(endBlock);
				rewriter.setInsertionPointToEnd(initBlock);
				mlir::cf::BranchOp::create(rewriter, op.getStep().getLoc(), iterator_next_block);

				rewriter.setInsertionPointToStart(iterator_next_block);

				mlir::emitpybytecode::ForIter::create(rewriter,
					op.getStep().getLoc(),
					iterator,
					&op.getStep().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());

				ASSERT(!op.getStep().empty());
				auto *iterator_exit_block = &op.getStep().back();
				ASSERT(iterator_exit_block->getTerminator());
				ASSERT(mlir::isa<mlir::py::BranchYieldOp>(iterator_exit_block->getTerminator()));

				rewriter.setInsertionPointToEnd(iterator_exit_block);
				rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
					iterator_exit_block->getTerminator(), &op.getBody().front());

				auto *for_iter_block = rewriter.createBlock(&op.getBody());
				mlir::emitpybytecode::ForIter::create(rewriter,
					op.getStep().getLoc(),
					iterator,
					&op.getStep().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());

				ASSERT(op.getStep().getArguments().size() == 1);
				rewriter.inlineRegionBefore(
					op.getStep(), *op->getParentRegion(), endBlock->getIterator());

				replace_loop_branch_yields(rewriter, op.getBody(), for_iter_block, endBlock);

				ASSERT(!op.getBody().empty());
				auto *body_exit_block = &op.getBody().back();
				ASSERT(body_exit_block->getTerminator());
				rewriter.inlineRegionBefore(
					op.getBody(), *op->getParentRegion(), endBlock->getIterator());

				replace_orelse_completion_yields(rewriter, op.getOrelse(), endBlock);
				rewriter.inlineRegionBefore(
					op.getOrelse(), *op->getParentRegion(), endBlock->getIterator());

				rewriter.eraseOp(op);


				return success();
			}
		};

		struct WhileOpLowering : public mlir::OpRewritePattern<mlir::py::WhileOp>
		{
			using OpRewritePattern<mlir::py::WhileOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::WhileOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				// See ForLoopOpLowering: innermost-first, so a nested loop's orelse
				// break/continue is in our region before we try to retarget it.
				if (has_pending_nested_orelse_control(op.getBody())) { return failure(); }

				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto &condition = op.getCondition();
				auto &condition_start = condition.getBlocks().front();
				ASSERT(!condition.getBlocks().empty());
				ASSERT(condition.back().getTerminator());

				auto condition_op =
					mlir::cast<mlir::py::ConditionOp>(condition.back().getTerminator());
				ASSERT(condition_op);

				rewriter.setInsertionPointToEnd(initBlock);
				mlir::cf::BranchOp::create(rewriter, condition_op.getLoc(), &condition_start);

				rewriter.setInsertionPoint(condition_op);
				auto should_jump = mlir::py::CastToBoolOp::create(
					rewriter, condition_op.getLoc(), rewriter.getI1Type(), condition_op.getCond());
				ASSERT(!op.getBody().empty());
				mlir::cf::CondBranchOp::create(rewriter,
					condition_op.getLoc(),
					should_jump,
					&op.getBody().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());
				rewriter.eraseOp(condition_op);
				rewriter.inlineRegionBefore(condition, endBlock);

				replace_loop_branch_yields(rewriter, op.getBody(), &condition_start, endBlock);

				rewriter.inlineRegionBefore(op.getBody(), endBlock);

				// Without this the orelse's completion yield survives region inlining
				// with no py.while parent left to satisfy its HasParent trait.
				replace_orelse_completion_yields(rewriter, op.getOrelse(), endBlock);
				rewriter.inlineRegionBefore(op.getOrelse(), endBlock);

				rewriter.eraseOp(op);

				return success();
			}
		};

		struct TryOpLowering : public mlir::OpRewritePattern<mlir::py::TryOp>
		{
			using OpRewritePattern<mlir::py::TryOp>::OpRewritePattern;

			template<typename FnT>
			void replace_controlflow_yield(mlir::Region &region, FnT &&callback) const
			{
				if (region.empty()) { return; }
				region.walk<WalkOrder::PreOrder>([callback](mlir::Operation *childOp) {
					if (is_flattened_region_op(childOp)) { return WalkResult::skip(); }
					if (mlir::isa<mlir::py::BranchYieldOp>(childOp)) {
						// Both normal-completion (kindless) and loop-control
						// (break/continue) yields are surfaced; the callback
						// dispatches on the kind.
						callback(childOp);
						return WalkResult::skip();
					}
					return WalkResult::advance();
				});
			}

			mlir::LogicalResult matchAndRewrite(mlir::py::TryOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				// Lower innermost-first when a finally is involved. A
				// break/continue nested in an inner try only surfaces into this
				// try's body once the inner try is lowered, and we must see it
				// before pre-scanning the loop-control kinds to thread through
				// our own finally (build_finally_loop_exits). MLIR's greedy
				// worklist doesn't guarantee inner-first, so defer (fail this
				// match) until any nested try in our regions has been lowered
				// away; the driver re-tries us when the inner try rewrites.
				// (Nested with/loops don't need this: with lowers in a later
				// pass, and a loop consumes its own break/continue.)
				if (!op.getFinally().empty()) {
					bool has_nested_try = false;
					auto scan = [&](mlir::Region &region) {
						if (has_nested_try || region.empty()) { return; }
						region.walk([&](mlir::py::TryOp) {
							has_nested_try = true;
							return WalkResult::interrupt();
						});
					};
					scan(op.getBody());
					for (mlir::Region &handler : op.getHandlers()) { scan(handler); }
					scan(op.getOrelse());
					if (has_nested_try) { return mlir::failure(); }
				}

				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto *body_start = &op.getBody().front();

				// Pre-build the per-kind finally exit paths for break/continue
				// while the finally region is still pristine (the loop below
				// rewrites it). Empty when there is no finally.
				const auto finally_exits = build_finally_loop_exits(rewriter, op, endBlock);

				replace_controlflow_yield(op.getBody(),
					[&rewriter, &op, &finally_exits, endBlock](mlir::Operation *childOp) {
						auto *current = childOp->getBlock();
						auto *next = rewriter.splitBlock(current, childOp->getIterator());
						rewriter.setInsertionPointToEnd(current);
						mlir::emitpybytecode::LeaveExceptionHandle::create(
							rewriter, childOp->getLoc());
						if (auto y = mlir::cast<mlir::py::BranchYieldOp>(childOp);
							y.getKind().has_value()) {
							// break/continue out of the try body: pop the
							// exception handler, then defer to the enclosing loop
							// (running the finally first if there is one).
							forward_loop_control_yield(rewriter, finally_exits, y);
							rewriter.eraseBlock(next);
							return;
						}
						if (op.getHandlers().empty()) {
							ASSERT(!op.getFinally().empty());
							mlir::cf::BranchOp::create(
								rewriter, childOp->getLoc(), &op.getFinally().front());
						} else if (!op.getOrelse().empty()) {
							mlir::cf::BranchOp::create(
								rewriter, childOp->getLoc(), &op.getOrelse().front());
						} else if (!op.getFinally().empty()) {
							mlir::cf::BranchOp::create(
								rewriter, childOp->getLoc(), &op.getFinally().front());
						} else {
							mlir::cf::BranchOp::create(rewriter, childOp->getLoc(), endBlock);
						}
						rewriter.eraseBlock(next);
					});
				rewriter.inlineRegionBefore(op.getBody(), endBlock);

				std::optional<mlir::IRMapping> finally_mapping;
				if (!op.getFinally().empty()) {
					finally_mapping = mlir::IRMapping{};

					rewriter.cloneRegionBefore(op.getFinally(),
						*endBlock->getParent(),
						endBlock->getIterator(),
						*finally_mapping);

					replace_controlflow_yield(op.getFinally(),
						[&rewriter, &op, &finally_mapping, endBlock](mlir::Operation *childOp) {
							// A break/continue written *inside* the finally overrides
							// whatever exit the try was heading for. Its kind drives
							// both the normal and the exceptional finally copies.
							auto kind_attr =
								mlir::cast<mlir::py::BranchYieldOp>(childOp).getKindAttr();
							// Normal-completion copy: kindless yields fall through to
							// the try's exit; an inside break/continue re-emits the
							// loop-control marker for the enclosing loop pass.
							{
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								if (kind_attr) {
									mlir::py::BranchYieldOp::create(
										rewriter, childOp->getLoc(), kind_attr);
								} else {
									mlir::cf::BranchOp::create(
										rewriter, childOp->getLoc(), endBlock);
								}
								rewriter.eraseBlock(next);
							}

							childOp = finally_mapping->lookup(childOp);
							ASSERT(childOp);
							// Exceptional copy: kindless yields re-raise the in-flight
							// exception after the finally; an inside break/continue
							// instead *swallows* it (Python semantics) and performs
							// the loop control flow.
							{
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								if (kind_attr) {
									mlir::emitpybytecode::ClearExceptionState::create(
										rewriter, childOp->getLoc());
									mlir::py::BranchYieldOp::create(
										rewriter, childOp->getLoc(), kind_attr);
								} else {
									mlir::emitpybytecode::ReRaiseOp::create(
										rewriter, childOp->getLoc(), endBlock);
								}
								rewriter.eraseBlock(next);
							}
						});
				}

				rewriter.setInsertionPointToEnd(initBlock);
				if (!op.getHandlers().empty()) {
					auto &handler = op.getHandlers().front();
					ASSERT(handler.getBlocks().size() == 1);
					auto handler_scope =
						mlir::cast<mlir::py::TryHandlerOp>(handler.front().getTerminator());
					ASSERT(handler_scope);
					mlir::emitpybytecode::SetupExceptionHandle::create(rewriter,
						op.getLoc(),
						body_start,
						handler_scope.getCond().empty() ? &handler_scope.getHandler().front()
														: &handler_scope.getCond().front());
				} else {
					ASSERT(finally_mapping.has_value());
					mlir::emitpybytecode::SetupExceptionHandle::create(rewriter,
						op.getLoc(),
						body_start,
						finally_mapping->lookup(&op.getFinally().front()));
				}

				if (!op.getHandlers().empty()) {
					for (auto e : llvm::enumerate(op.getHandlers().drop_back())) {
						auto &handler = e.value();
						auto idx = e.index();

						ASSERT(handler.getBlocks().size() == 1);
						auto handler_scope =
							mlir::cast<mlir::py::TryHandlerOp>(handler.front().getTerminator());
						ASSERT(handler_scope);

						if (!handler_scope.getCond().empty()) {
							auto cond = mlir::cast<mlir::py::ConditionOp>(
								handler_scope.getCond().back().getTerminator());
							ASSERT(cond);
							rewriter.setInsertionPoint(cond);
							auto &next_handler = op.getHandlers()[idx + 1];
							ASSERT(next_handler.getBlocks().size() == 1);
							auto next_handler_scope = mlir::cast<mlir::py::TryHandlerOp>(
								next_handler.front().getTerminator());
							ASSERT(next_handler_scope);

							rewriter.replaceOpWithNewOp<mlir::py::CondBranchSubclassOp>(cond,
								cond.getCond(),
								mlir::ValueRange{},
								mlir::ValueRange{},
								next_handler_scope.getCond().empty()
									? &next_handler_scope.getHandler().front()
									: &next_handler_scope.getCond().front(),
								&handler_scope.getHandler().front());
							rewriter.inlineRegionBefore(handler_scope.getCond(), endBlock);
						}
						replace_controlflow_yield(handler_scope.getHandler(),
							[&rewriter, &op, &finally_exits, endBlock](mlir::Operation *childOp) {
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								mlir::emitpybytecode::ClearExceptionState::create(
									rewriter, op.getLoc());
								if (auto y = mlir::cast<mlir::py::BranchYieldOp>(childOp);
									y.getKind().has_value()) {
									// break/continue out of an except handler:
									// clear the active exception, then defer to
									// the enclosing loop.
									forward_loop_control_yield(rewriter, finally_exits, y);
									rewriter.eraseBlock(next);
									return;
								}
								if (!op.getFinally().empty()) {
									mlir::cf::BranchOp::create(
										rewriter, childOp->getLoc(), &op.getFinally().front());
								} else {
									mlir::cf::BranchOp::create(
										rewriter, childOp->getLoc(), endBlock);
								}
								rewriter.eraseBlock(next);
							});
						rewriter.inlineRegionBefore(handler_scope.getHandler(), endBlock);
					}

					{
						auto &handler = op.getHandlers().back();
						ASSERT(handler.getBlocks().size() == 1);
						auto handler_scope =
							mlir::cast<mlir::py::TryHandlerOp>(handler.front().getTerminator());
						ASSERT(handler_scope);
						if (!handler_scope.getCond().empty()) {
							auto cond = mlir::cast<mlir::py::ConditionOp>(
								handler_scope.getCond().back().getTerminator());
							ASSERT(cond);

							auto *reraise_block = rewriter.createBlock(&handler_scope.getCond());
							mlir::py::RaiseOp::create(rewriter, cond.getLoc());

							rewriter.setInsertionPoint(cond);
							rewriter.replaceOpWithNewOp<mlir::py::CondBranchSubclassOp>(cond,
								cond.getCond(),
								mlir::ValueRange{},
								mlir::ValueRange{},
								op.getFinally().empty()
									? reraise_block
									: finally_mapping->lookup(&op.getFinally().front()),
								&handler_scope.getHandler().front());

							rewriter.inlineRegionBefore(handler_scope.getCond(), endBlock);
						}

						replace_controlflow_yield(handler_scope.getHandler(),
							[&rewriter, &op, &finally_exits, endBlock](mlir::Operation *childOp) {
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								mlir::emitpybytecode::ClearExceptionState::create(
									rewriter, op.getLoc());
								if (auto y = mlir::cast<mlir::py::BranchYieldOp>(childOp);
									y.getKind().has_value()) {
									// break/continue out of an except handler:
									// clear the active exception, then defer to
									// the enclosing loop.
									forward_loop_control_yield(rewriter, finally_exits, y);
									rewriter.eraseBlock(next);
									return;
								}
								if (!op.getFinally().empty()) {
									mlir::cf::BranchOp::create(
										rewriter, childOp->getLoc(), &op.getFinally().front());
								} else {
									mlir::cf::BranchOp::create(
										rewriter, childOp->getLoc(), endBlock);
								}
								rewriter.eraseBlock(next);
							});
						rewriter.inlineRegionBefore(handler_scope.getHandler(), endBlock);
					}
				}

				replace_controlflow_yield(op.getOrelse(),
					[&rewriter, &op, &finally_exits, endBlock](mlir::Operation *childOp) {
						auto *current = childOp->getBlock();
						auto *next = rewriter.splitBlock(current, childOp->getIterator());
						rewriter.setInsertionPointToEnd(current);
						if (auto y = mlir::cast<mlir::py::BranchYieldOp>(childOp);
							y.getKind().has_value()) {
							// break/continue out of the else clause: the handler
							// was already left when the body completed normally,
							// so just defer to the enclosing loop.
							forward_loop_control_yield(rewriter, finally_exits, y);
							rewriter.eraseBlock(next);
							return;
						}
						if (!op.getFinally().empty()) {
							mlir::cf::BranchOp::create(
								rewriter, childOp->getLoc(), &op.getFinally().front());
						} else {
							mlir::cf::BranchOp::create(rewriter, childOp->getLoc(), endBlock);
						}
						rewriter.eraseBlock(next);
					});
				rewriter.inlineRegionBefore(op.getOrelse(), endBlock);

				rewriter.inlineRegionBefore(op.getFinally(), endBlock);

				rewriter.eraseOp(op);

				return success();
			}
		};

		struct WithOpLowering : public mlir::OpRewritePattern<mlir::py::WithOp>
		{
			using OpRewritePattern<mlir::py::WithOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::WithOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto *body_start = &op.getBody().front();
				auto *cleanup_block = rewriter.createBlock(endBlock);
				auto *exit_block = rewriter.createBlock(endBlock);

				// Emits the non-exceptional __exit__(None, None, None) sequence
				// at the current insertion point. Shared by the normal-exit path
				// and the break/continue path (both leave without an exception).
				auto emit_normal_exit = [&rewriter, &op]() {
					for (const auto &item : op.getItems()) {
						auto exit = mlir::py::LoadMethodOp::create(rewriter,
							item.getLoc(),
							mlir::py::PyObjectType::get(rewriter.getContext()),
							item,
							"__exit__");
						auto none = mlir::py::ConstantOp::create(
							rewriter, item.getLoc(), rewriter.getNoneType());
						mlir::py::FunctionCallOp::create(rewriter,
							item.getLoc(),
							mlir::py::PyObjectType::get(rewriter.getContext()),
							exit,
							std::vector<mlir::Value>{ none, none, none },
							mlir::DenseStringElementsAttr::get(
								mlir::VectorType::get(
									{ 0 }, mlir::StringAttr::get(rewriter.getContext()).getType()),
								{}),
							std::vector<mlir::Value>{},
							false,
							false);
						mlir::py::ClearExceptionStateOp::create(rewriter, item.getLoc());
					}
				};

				op.getBody().walk<WalkOrder::PreOrder>(
					[&rewriter, exit_block, cleanup_block, endBlock, &emit_normal_exit](
						mlir::Operation *childOp) {
						if (is_flattened_region_op(childOp)) { return WalkResult::skip(); }
						if (auto op = mlir::dyn_cast<mlir::py::RaiseOp>(childOp)) {
							rewriter.setInsertionPoint(op);
							if (op.getCause()) {
								rewriter.replaceOpWithNewOp<mlir::emitpybytecode::RaiseVarargs>(op,
									op.getException(),
									op.getCause(),
									BlockRange{ cleanup_block });
							} else if (op.getException()) {
								rewriter.replaceOpWithNewOp<mlir::emitpybytecode::RaiseVarargs>(
									op, op.getException(), nullptr, BlockRange{ cleanup_block });
							} else {
								rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ReRaiseOp>(
									op, BlockRange{ cleanup_block });
							}
						} else if (auto y = mlir::dyn_cast<mlir::py::BranchYieldOp>(childOp);
							y && !y.getKind().has_value()) {
							auto *current = y->getBlock();
							auto *next = rewriter.splitBlock(current, y->getIterator());
							rewriter.setInsertionPointToEnd(current);
							mlir::emitpybytecode::LeaveExceptionHandle::create(
								rewriter, y->getLoc());
							mlir::cf::BranchOp::create(rewriter, y->getLoc(), exit_block);
							rewriter.eraseBlock(next);
						} else if (auto y = mlir::dyn_cast<mlir::py::BranchYieldOp>(childOp);
							y && y.getKind().has_value()) {
							// break/continue out of the with body: leave the
							// exception handler, run __exit__, then hand the marker
							// to the enclosing loop on a dedicated exit path.
							auto *current = y->getBlock();
							auto *next = rewriter.splitBlock(current, y->getIterator());
							auto *lc_block = rewriter.createBlock(endBlock);
							rewriter.setInsertionPointToEnd(current);
							mlir::emitpybytecode::LeaveExceptionHandle::create(
								rewriter, y->getLoc());
							mlir::cf::BranchOp::create(rewriter, y->getLoc(), lc_block);
							rewriter.setInsertionPointToStart(lc_block);
							emit_normal_exit();
							mlir::py::BranchYieldOp::create(rewriter, y->getLoc(), y.getKindAttr());
							rewriter.eraseBlock(next);
						}
						return WalkResult::advance();
					});

				rewriter.inlineRegionBefore(op.getBody(), endBlock);

				// Multi-item with-statements (with a, b, c: ...) are not
				// yet supported end-to-end: MLIRGenerator currently TODOs
				// out for items().size() > 1, so the dialect op only ever
				// arrives here with a single item. The loops below over
				// op.getItems() exist for shape symmetry with the future
				// multi-item version but bail explicitly until that work
				// lands.
				ASSERT(op.getItems().size() == 1
					   && "WithOp lowering does not yet support multiple context managers");
				rewriter.setInsertionPointToStart(cleanup_block);
				for (const auto &item : op.getItems()) {
					auto exit = mlir::py::LoadMethodOp::create(rewriter,
						item.getLoc(),
						mlir::py::PyObjectType::get(rewriter.getContext()),
						item,
						"__exit__");

					auto except_result = mlir::py::WithExceptStartOp::create(rewriter,
						item.getLoc(),
						mlir::py::PyObjectType::get(rewriter.getContext()),
						exit);

					auto *reraise_block = rewriter.createBlock(endBlock);
					auto *continue_block = rewriter.createBlock(endBlock);
					rewriter.setInsertionPointAfter(except_result);

					auto cond = mlir::py::CastToBoolOp::create(
						rewriter, except_result.getLoc(), rewriter.getI1Type(), except_result);
					mlir::cf::CondBranchOp::create(
						rewriter, cond.getLoc(), cond, continue_block, reraise_block);

					rewriter.setInsertionPointToStart(reraise_block);
					mlir::emitpybytecode::ReRaiseOp::create(rewriter, item.getLoc(), endBlock);

					rewriter.setInsertionPointToStart(continue_block);
					mlir::emitpybytecode::ClearExceptionState::create(rewriter, item.getLoc());
					mlir::cf::BranchOp::create(rewriter, op.getLoc(), endBlock);
				}

				rewriter.setInsertionPointToStart(exit_block);
				emit_normal_exit();
				mlir::cf::BranchOp::create(rewriter, op.getLoc(), endBlock);

				rewriter.setInsertionPointToEnd(initBlock);
				mlir::emitpybytecode::SetupWith::create(
					rewriter, op.getLoc(), body_start, cleanup_block);

				rewriter.eraseOp(op);

				return success();
			}
		};


		struct PythonToPythonBytecodePass
			: public PassWrapper<PythonToPythonBytecodePass, OperationPass<ModuleOp>>
		{
			MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PythonToPythonBytecodePass)

			void getDependentDialects(DialectRegistry &registry) const override
			{
				registry.insert<PythonDialect, emitpybytecode::EmitPythonBytecodeDialect>();
			}

			StringRef getArgument() const final { return "python-to-pythonbytecode"; }

			void runOnOperation() final;
		};

		// Pass scaffolds for the four region-bearing control-flow ops.
		// Each pass applies its single lowering pattern greedily on the
		// module. Dialect dependencies match PythonToPythonBytecodePass's
		// (Python source dialect + EmitPythonBytecode target dialect); the
		// patterns also create cf::BranchOp / func::FuncOp internally, but
		// those dialects are already loaded by the time the pipeline runs.
		template<typename Derived, const char *Argument, typename... Patterns>
		struct PatternConversionPass : public PassWrapper<Derived, OperationPass<ModuleOp>>
		{
			void getDependentDialects(DialectRegistry &registry) const override
			{
				registry.insert<PythonDialect, emitpybytecode::EmitPythonBytecodeDialect>();
			}

			StringRef getArgument() const final { return Argument; }

			void runOnOperation() final
			{
				mlir::RewritePatternSet patterns(&this->getContext());
				patterns.template add<Patterns...>(&this->getContext());

				GreedyRewriteConfig config;
				config.setStrictness(GreedyRewriteStrictness::AnyOp);
				config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
				config.setUseTopDownTraversal(true);
				FrozenRewritePatternSet frozen{ std::move(patterns) };

				(void)applyPatternsGreedily(this->getOperation(), frozen, config);
			}
		};

		template<typename Derived, typename Pattern, const char *Argument>
		using SinglePatternConversionPass = PatternConversionPass<Derived, Argument, Pattern>;

		inline constexpr char kConvertLoopsArg[] = "convert-py-loops";
		inline constexpr char kConvertForLoopArg[] = "convert-py-forloop";
		inline constexpr char kConvertWhileLoopArg[] = "convert-py-while";
		inline constexpr char kConvertTryArg[] = "convert-py-try";
		inline constexpr char kConvertWithArg[] = "convert-py-with";

		// Both loop patterns must share one greedy driver. A break/continue in a
		// nested loop's orelse binds to the enclosing loop and can only be retargeted
		// after the nested loop is flattened, which the patterns arrange by deferring
		// (see has_pending_nested_orelse_control). Deferral can only resolve if the
		// pattern being waited on is available in the same run: with `for` and `while`
		// in separate passes, a `for` containing a `while ... else: continue` would
		// wait for a pattern that does not run until the next pass, and never lower.
		//
		// The single-pattern passes below stay registered for python-mlir-opt, where
		// running one lowering in isolation is what the Conversion lit tests want.
		struct ConvertLoopsPass
			: public PatternConversionPass<ConvertLoopsPass,
				  kConvertLoopsArg,
				  ForLoopOpLowering,
				  WhileOpLowering>
		{
			MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertLoopsPass)
		};

		struct ConvertForLoopPass
			: public SinglePatternConversionPass<ConvertForLoopPass,
				  ForLoopOpLowering,
				  kConvertForLoopArg>
		{
			MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertForLoopPass)
		};

		struct ConvertWhileLoopPass
			: public SinglePatternConversionPass<ConvertWhileLoopPass,
				  WhileOpLowering,
				  kConvertWhileLoopArg>
		{
			MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertWhileLoopPass)
		};

		struct ConvertTryPass
			: public SinglePatternConversionPass<ConvertTryPass, TryOpLowering, kConvertTryArg>
		{
			MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertTryPass)
		};

		struct ConvertWithPass
			: public SinglePatternConversionPass<ConvertWithPass, WithOpLowering, kConvertWithArg>
		{
			MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertWithPass)
		};

		// Pattern: rewrite a zero-operand func.return by inserting a None
		// constant operand and, if RemoveDeadValues also rewrote the
		// parent FuncOp's signature to return nothing, restoring its
		// declared result type to PyObjectType. The bytecode emitter
		// assumes every function returns a value (Python's "every
		// function returns at minimum None") regardless of whether MLIR
		// sees the result as used.
		//
		// Reaches for emitpybytecode::ConstantOp because the pass runs
		// after PythonToPythonBytecodePass has already lowered
		// py.constant; using py.constant here would re-introduce an
		// illegal source-dialect op into the lowered IR.
		struct MaterialiseReturnNonePattern : public mlir::OpRewritePattern<mlir::func::ReturnOp>
		{
			using mlir::OpRewritePattern<mlir::func::ReturnOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				if (op.getNumOperands() != 0) { return mlir::failure(); }
				auto parent = op->getParentOfType<mlir::func::FuncOp>();
				if (!parent) { return mlir::failure(); }
				auto pyobject_ty = mlir::py::PyObjectType::get(rewriter.getContext());
				rewriter.setInsertionPoint(op);
				auto none = mlir::emitpybytecode::ConstantOp::create(
					rewriter, op.getLoc(), pyobject_ty, rewriter.getUnitAttr());
				rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, mlir::ValueRange{ none });

				// Restore the function signature if RemoveDeadValues stripped
				// the result type. Plain assignment to the function-type
				// attribute is fine here because the parent op's properties
				// aren't tracked by the pattern rewriter's mutation tracking
				// (we already produced a successful match-and-rewrite via
				// replaceOpWithNewOp above).
				if (parent.getFunctionType().getNumResults() == 0) {
					auto fn_ty = parent.getFunctionType();
					parent.setFunctionType(rewriter.getFunctionType(
						fn_ty.getInputs(), mlir::TypeRange{ pyobject_ty }));
				}
				return mlir::success();
			}
		};

		struct MaterialiseReturnNonePass
			: public PassWrapper<MaterialiseReturnNonePass, OperationPass<mlir::func::FuncOp>>
		{
			MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterialiseReturnNonePass)

			void getDependentDialects(DialectRegistry &registry) const override
			{
				registry.insert<emitpybytecode::EmitPythonBytecodeDialect>();
			}

			StringRef getArgument() const final { return "materialise-return-none"; }

			void runOnOperation() final
			{
				mlir::RewritePatternSet patterns(&getContext());
				patterns.add<MaterialiseReturnNonePattern>(&getContext());

				GreedyRewriteConfig config;
				config.setStrictness(GreedyRewriteStrictness::AnyOp);
				FrozenRewritePatternSet frozen{ std::move(patterns) };

				(void)applyPatternsGreedily(getOperation(), frozen, config);
			}
		};
	}// namespace

	void PythonToPythonBytecodePass::runOnOperation()
	{
		mlir::RewritePatternSet patterns(&getContext());
		populateArithPatterns(patterns);
		populateAttributeSubscriptPatterns(patterns);
		populateCollectionPatterns(patterns);
		populateControlFlowPatterns(patterns);
		populateFunctionPatterns(patterns);
		populateImportPatterns(patterns);
		populateLoadStorePatterns(patterns);
		// ForLoop / While / Try / With lowerings remain in this file but
		// run in dedicated passes (ConvertPyForLoop / While / Try / With)
		// ahead of this monolithic conversion pass, so canonicalize / CSE
		// can simplify between their structural rewrites.

		GreedyRewriteConfig config;
		config.setStrictness(GreedyRewriteStrictness::AnyOp);
		config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
		config.setUseTopDownTraversal(true);
		FrozenRewritePatternSet frozen_patterns{ std::move(patterns) };

		// applyPatternsGreedily returns failure() when the driver hits
		// its iteration limit without reaching a fixed point. The
		// remaining work is to figure out which pattern keeps firing
		// (likely one that always replaces-with-itself in some edge
		// case) and either fix it or change the pass to use full
		// dialect conversion. For now the IR is verified after the
		// pass runs (PassManager's default), so a real failure would
		// surface there; treating the rewriter's return as
		// signalPassFailure() would be a false positive today.
		(void)applyPatternsGreedily(getOperation(), frozen_patterns, config);
	}

	std::unique_ptr<Pass> createPythonToPythonBytecodePass()
	{
		return std::make_unique<PythonToPythonBytecodePass>();
	}

	std::unique_ptr<Pass> createConvertForLoopPass()
	{
		return std::make_unique<ConvertForLoopPass>();
	}

	std::unique_ptr<Pass> createConvertWhileLoopPass()
	{
		return std::make_unique<ConvertWhileLoopPass>();
	}

	std::unique_ptr<Pass> createConvertLoopsPass() { return std::make_unique<ConvertLoopsPass>(); }

	std::unique_ptr<Pass> createConvertTryPass() { return std::make_unique<ConvertTryPass>(); }

	std::unique_ptr<Pass> createConvertWithPass() { return std::make_unique<ConvertWithPass>(); }

	std::unique_ptr<Pass> createMaterialiseReturnNonePass()
	{
		return std::make_unique<MaterialiseReturnNonePass>();
	}

}// namespace py
}// namespace mlir
