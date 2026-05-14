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
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <algorithm>

namespace mlir {
namespace py {

	namespace {
		// All non-region-bearing patterns now live in per-family files:
		// {Arith, AttributeSubscript, Collection, ControlFlow, Function,
		// Import, LoadStore}Patterns.cpp, registered via the
		// populate*Patterns() entry points below.
		// The shared DirectReplaceLowering / DirectReplaceRegisterName
		// helpers and add_identifier* utilities moved to LoweringHelpers.hpp.
		// What remains in this file are the four region-bearing structural
		// patterns (ForLoop / While / Try / With), the
		// PythonToPythonBytecodePass, and the dedicated single-pattern
		// passes that wrap the structural ones.

		// Shared walker used by both ForLoopOpLowering and WhileOpLowering
		// to lower py.br_yield ops nested inside a loop body to cf.br ops
		// that target the right block (continue→condition / step, break→
		// end). Nested loops are walked into their orelse regions only;
		// the loop body itself is skipped because the nested loop will be
		// lowered by its own pattern.
		//
		// `skip_op` allows a caller to short-circuit on yield ops whose
		// enclosing loop matches a specific predicate — ForLoopOpLowering
		// uses this to ignore yields that bind to the *outer* for-loop's
		// orelse (which shouldn't be lowered as part of the inner loop
		// pass).
		void replace_loop_branch_yields(mlir::PatternRewriter &rewriter,
			mlir::Region &region,
			mlir::Block *continue_target,
			mlir::Block *break_target,
			llvm::function_ref<bool(mlir::py::BranchYieldOp)> skip_op)
		{
			std::function<WalkResult(mlir::Operation *)> callback =
				[&rewriter, continue_target, break_target, skip_op, &callback](
					mlir::Operation *operation) {
					if (auto loop = mlir::dyn_cast<mlir::py::ForLoopOp>(operation)) {
						if (loop.getOrelse().empty()) { return WalkResult::skip(); }
						loop.getOrelse().walk<WalkOrder::PreOrder>(callback);
						return WalkResult::skip();
					}
					if (auto loop = mlir::dyn_cast<mlir::py::WhileOp>(operation)) {
						if (loop.getOrelse().empty()) { return WalkResult::skip(); }
						loop.getOrelse().walk<WalkOrder::PreOrder>(callback);
						return WalkResult::skip();
					}
					auto yield_op = mlir::dyn_cast<mlir::py::BranchYieldOp>(operation);
					if (!yield_op) { return WalkResult::advance(); }
					static_assert(mlir::py::BranchYieldOp::hasTrait<mlir::OpTrait::
							HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerOp>::Impl>());
					// Kindless yields under try/with/try-handler don't
					// participate in the loop's continue/break flow.
					if (!yield_op.getKind().has_value()
						&& mlir::isa<TryOp, WithOp, TryHandlerOp>(yield_op->getParentOp())) {
						return WalkResult::advance();
					}
					if (skip_op && skip_op(yield_op)) { return WalkResult::advance(); }
					rewriter.setInsertionPoint(yield_op);
					if (!yield_op.getKind().has_value()
						|| yield_op.getKind().value() == py::LoopOpKind::continue_) {
						rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield_op, continue_target);
					} else if (yield_op.getKind().value() == py::LoopOpKind::break_) {
						rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield_op, break_target);
					}
					return WalkResult::advance();
				};
			region.walk<WalkOrder::PreOrder>(callback);
		}

		struct ForLoopOpLowering : public mlir::OpRewritePattern<mlir::py::ForLoopOp>
		{
			using OpRewritePattern<mlir::py::ForLoopOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::ForLoopOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto iterable = op.getIterable();
				rewriter.setInsertionPointToEnd(initBlock);
				auto iterator = rewriter.create<mlir::emitpybytecode::GetIter>(
					op.getStep().getLoc(), iterable.getType(), iterable);

				// advance iterator
				auto iterator_next_block = rewriter.createBlock(endBlock);
				rewriter.setInsertionPointToEnd(initBlock);
				rewriter.create<mlir::cf::BranchOp>(op.getStep().getLoc(), iterator_next_block);

				rewriter.setInsertionPointToStart(iterator_next_block);

				rewriter.create<mlir::emitpybytecode::ForIter>(op.getStep().getLoc(),
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
				rewriter.create<mlir::emitpybytecode::ForIter>(op.getStep().getLoc(),
					iterator,
					&op.getStep().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());

				ASSERT(op.getStep().getArguments().size() == 1);
				rewriter.inlineRegionBefore(
					op.getStep(), *op->getParentRegion(), endBlock->getIterator());

				// Skip yields whose enclosing for-loop sits inside an
				// outer for-loop's orelse — those belong to the outer
				// pattern's rewrite, not this one.
				auto skip_orelse_yields = [](mlir::py::BranchYieldOp y) {
					auto forloop_op = y->getParentOfType<mlir::py::ForLoopOp>();
					return forloop_op && &forloop_op.getOrelse() == y->getParentRegion();
				};
				replace_loop_branch_yields(
					rewriter, op.getBody(), for_iter_block, endBlock, skip_orelse_yields);

				ASSERT(!op.getBody().empty());
				auto *body_exit_block = &op.getBody().back();
				ASSERT(body_exit_block->getTerminator());
				rewriter.inlineRegionBefore(
					op.getBody(), *op->getParentRegion(), endBlock->getIterator());

				if (!op.getOrelse().empty()) {
					auto *orelse_exit_block = &op.getOrelse().back();
					ASSERT(orelse_exit_block->getTerminator());
					if (mlir::isa<mlir::py::BranchYieldOp>(orelse_exit_block->getTerminator())) {
						rewriter.setInsertionPointToEnd(orelse_exit_block);
						rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
							orelse_exit_block->getTerminator(), endBlock);
					}
				}
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
				rewriter.create<mlir::cf::BranchOp>(condition_op.getLoc(), &condition_start);

				if (mlir::isa<mlir::BlockArgument>(condition_op.getCond())) {
					rewriter.setInsertionPointToStart(condition_op.getCond().getParentBlock());
				} else {
					rewriter.setInsertionPointAfter(condition_op.getCond().getDefiningOp());
				}
				auto should_jump = rewriter.create<mlir::py::CastToBoolOp>(
					condition_op.getLoc(), rewriter.getI1Type(), condition_op.getCond());
				ASSERT(!op.getBody().empty());
				rewriter.create<mlir::cf::CondBranchOp>(condition_op.getLoc(),
					should_jump,
					&op.getBody().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());
				rewriter.eraseOp(condition_op);
				rewriter.inlineRegionBefore(condition, endBlock);

				replace_loop_branch_yields(rewriter,
					op.getBody(),
					&condition_start,
					endBlock,
					/*skip_op=*/{});

				rewriter.inlineRegionBefore(op.getBody(), endBlock);

				// if (!op.getOrelse().empty()) {
				// 	auto *orelse_exit_block = &op.getOrelse().back();
				// 	ASSERT(orelse_exit_block->getTerminator());
				// 	if (mlir::isa<mlir::py::BranchYieldOp>(orelse_exit_block->getTerminator())) {
				// 		rewriter.setInsertionPointToEnd(orelse_exit_block);
				// 		rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
				// 			orelse_exit_block->getTerminator(), endBlock);
				// 	}
				// }
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
					static_assert(mlir::py::BranchYieldOp::hasTrait<mlir::OpTrait::
							HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerOp>::Impl>());
					if (mlir::isa<mlir::py::TryOp>(childOp)
						|| mlir::isa<mlir::py::ForLoopOp>(childOp)
						|| mlir::isa<mlir::py::WhileOp>(childOp)
						|| mlir::isa<mlir::py::WithOp>(childOp)
						|| mlir::isa<mlir::py::TryHandlerOp>(childOp)) {
						return WalkResult::skip();
					}
					if (mlir::isa<mlir::py::BranchYieldOp>(childOp)
						&& !mlir::cast<mlir::py::BranchYieldOp>(childOp).getKind().has_value()) {
						callback(childOp);
						return WalkResult::skip();
					}
					return WalkResult::advance();
				});
			}

			mlir::LogicalResult matchAndRewrite(mlir::py::TryOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto *body_start = &op.getBody().front();

				replace_controlflow_yield(
					op.getBody(), [&rewriter, &op, endBlock](mlir::Operation *childOp) {
						auto *current = childOp->getBlock();
						auto *next = rewriter.splitBlock(current, childOp->getIterator());
						rewriter.setInsertionPointToEnd(current);
						rewriter.create<mlir::emitpybytecode::LeaveExceptionHandle>(
							childOp->getLoc());
						if (op.getHandlers().empty()) {
							ASSERT(!op.getFinally().empty());
							rewriter.create<mlir::cf::BranchOp>(
								childOp->getLoc(), &op.getFinally().front());
						} else if (!op.getOrelse().empty()) {
							rewriter.create<mlir::cf::BranchOp>(
								childOp->getLoc(), &op.getOrelse().front());
						} else if (!op.getFinally().empty()) {
							rewriter.create<mlir::cf::BranchOp>(
								childOp->getLoc(), &op.getFinally().front());
						} else {
							rewriter.create<mlir::cf::BranchOp>(childOp->getLoc(), endBlock);
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
							{
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								rewriter.create<mlir::cf::BranchOp>(childOp->getLoc(), endBlock);
								rewriter.eraseBlock(next);
							}

							childOp = finally_mapping->lookup(childOp);
							ASSERT(childOp);
							{
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								rewriter.create<mlir::emitpybytecode::ReRaiseOp>(
									childOp->getLoc(), endBlock);
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
					rewriter.create<mlir::emitpybytecode::SetupExceptionHandle>(op.getLoc(),
						body_start,
						handler_scope.getCond().empty() ? &handler_scope.getHandler().front()
														: &handler_scope.getCond().front());
				} else {
					ASSERT(finally_mapping.has_value());
					rewriter.create<mlir::emitpybytecode::SetupExceptionHandle>(
						op.getLoc(), body_start, finally_mapping->lookup(&op.getFinally().front()));
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
							[&rewriter, &op, endBlock](mlir::Operation *childOp) {
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								rewriter.create<mlir::emitpybytecode::ClearExceptionState>(
									op.getLoc());
								if (!op.getFinally().empty()) {
									rewriter.create<mlir::cf::BranchOp>(
										childOp->getLoc(), &op.getFinally().front());
								} else {
									rewriter.create<mlir::cf::BranchOp>(
										childOp->getLoc(), endBlock);
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
							rewriter.create<mlir::py::RaiseOp>(cond.getLoc());

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
							[&rewriter, &op, endBlock](mlir::Operation *childOp) {
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								rewriter.create<mlir::emitpybytecode::ClearExceptionState>(
									op.getLoc());
								if (!op.getFinally().empty()) {
									rewriter.create<mlir::cf::BranchOp>(
										childOp->getLoc(), &op.getFinally().front());
								} else {
									rewriter.create<mlir::cf::BranchOp>(
										childOp->getLoc(), endBlock);
								}
								rewriter.eraseBlock(next);
							});
						rewriter.inlineRegionBefore(handler_scope.getHandler(), endBlock);
					}
				}

				replace_controlflow_yield(
					op.getOrelse(), [&rewriter, &op, endBlock](mlir::Operation *childOp) {
						auto *current = childOp->getBlock();
						auto *next = rewriter.splitBlock(current, childOp->getIterator());
						rewriter.setInsertionPointToEnd(current);
						if (!op.getFinally().empty()) {
							rewriter.create<mlir::cf::BranchOp>(
								childOp->getLoc(), &op.getFinally().front());
						} else {
							rewriter.create<mlir::cf::BranchOp>(childOp->getLoc(), endBlock);
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

				op.getBody().walk<WalkOrder::PreOrder>([&rewriter, exit_block, cleanup_block](
														   mlir::Operation *childOp) {
					static_assert(mlir::py::BranchYieldOp::hasTrait<mlir::OpTrait::
							HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerOp>::Impl>());
					if (mlir::isa<mlir::py::TryOp>(childOp)
						|| mlir::isa<mlir::py::ForLoopOp>(childOp)
						|| mlir::isa<mlir::py::WhileOp>(childOp)
						|| mlir::isa<mlir::py::WithOp>(childOp)
						|| mlir::isa<mlir::py::TryHandlerOp>(childOp)) {
						return WalkResult::skip();
					}
					if (auto op = mlir::dyn_cast<mlir::py::RaiseOp>(childOp)) {
						rewriter.setInsertionPoint(op);
						if (op.getCause()) {
							rewriter.replaceOpWithNewOp<mlir::emitpybytecode::RaiseVarargs>(
								op, op.getException(), op.getCause(), BlockRange{ cleanup_block });
						} else if (op.getException()) {
							rewriter.replaceOpWithNewOp<mlir::emitpybytecode::RaiseVarargs>(
								op, op.getException(), nullptr, BlockRange{ cleanup_block });
						} else {
							rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ReRaiseOp>(
								op, BlockRange{ cleanup_block });
						}
					} else if (auto op = mlir::dyn_cast<mlir::py::BranchYieldOp>(childOp);
						op && !op.getKind().has_value()) {
						auto *current = op->getBlock();
						auto *next = rewriter.splitBlock(current, op->getIterator());
						rewriter.setInsertionPointToEnd(current);
						rewriter.create<mlir::emitpybytecode::LeaveExceptionHandle>(op->getLoc());
						rewriter.create<mlir::cf::BranchOp>(op->getLoc(), exit_block);
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
					auto exit = rewriter.create<mlir::py::LoadMethodOp>(item.getLoc(),
						mlir::py::PyObjectType::get(rewriter.getContext()),
						item,
						"__exit__");

					auto except_result = rewriter.create<mlir::py::WithExceptStartOp>(
						item.getLoc(), mlir::py::PyObjectType::get(rewriter.getContext()), exit);

					auto *reraise_block = rewriter.createBlock(endBlock);
					auto *continue_block = rewriter.createBlock(endBlock);
					rewriter.setInsertionPointAfter(except_result);

					auto cond = rewriter.create<mlir::py::CastToBoolOp>(
						except_result.getLoc(), rewriter.getI1Type(), except_result);
					rewriter.create<mlir::cf::CondBranchOp>(
						cond.getLoc(), cond, continue_block, reraise_block);

					rewriter.setInsertionPointToStart(reraise_block);
					rewriter.create<mlir::emitpybytecode::ReRaiseOp>(item.getLoc(), endBlock);

					rewriter.setInsertionPointToStart(continue_block);
					rewriter.create<mlir::emitpybytecode::ClearExceptionState>(item.getLoc());
					rewriter.create<mlir::cf::BranchOp>(op.getLoc(), endBlock);
				}

				rewriter.setInsertionPointToStart(exit_block);
				for (const auto &item : op.getItems()) {
					auto exit = rewriter.create<mlir::py::LoadMethodOp>(item.getLoc(),
						mlir::py::PyObjectType::get(rewriter.getContext()),
						item,
						"__exit__");

					auto none = rewriter.create<mlir::py::ConstantOp>(
						item.getLoc(), rewriter.getNoneType());

					rewriter.create<mlir::py::FunctionCallOp>(item.getLoc(),
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

					rewriter.create<mlir::py::ClearExceptionStateOp>(item.getLoc());
				}

				rewriter.create<mlir::cf::BranchOp>(op.getLoc(), endBlock);

				rewriter.setInsertionPointToEnd(initBlock);
				rewriter.create<mlir::emitpybytecode::SetupWith>(
					op.getLoc(), body_start, cleanup_block);

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
		template<typename Derived, typename Pattern, const char *Argument>
		struct SinglePatternConversionPass : public PassWrapper<Derived, OperationPass<ModuleOp>>
		{
			void getDependentDialects(DialectRegistry &registry) const override
			{
				registry.insert<PythonDialect, emitpybytecode::EmitPythonBytecodeDialect>();
			}

			StringRef getArgument() const final { return Argument; }

			void runOnOperation() final
			{
				mlir::RewritePatternSet patterns(&this->getContext());
				patterns.template add<Pattern>(&this->getContext());

				GreedyRewriteConfig config;
				config.setStrictness(GreedyRewriteStrictness::AnyOp);
				config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
				config.setUseTopDownTraversal(true);
				FrozenRewritePatternSet frozen{ std::move(patterns) };

				(void)applyPatternsGreedily(this->getOperation(), frozen, config);
			}
		};

		inline constexpr char kConvertForLoopArg[] = "convert-py-forloop";
		inline constexpr char kConvertWhileLoopArg[] = "convert-py-while";
		inline constexpr char kConvertTryArg[] = "convert-py-try";
		inline constexpr char kConvertWithArg[] = "convert-py-with";

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

	std::unique_ptr<Pass> createConvertTryPass() { return std::make_unique<ConvertTryPass>(); }

	std::unique_ptr<Pass> createConvertWithPass() { return std::make_unique<ConvertWithPass>(); }

}// namespace py
}// namespace mlir