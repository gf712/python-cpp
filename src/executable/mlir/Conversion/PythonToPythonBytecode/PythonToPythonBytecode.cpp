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
		namespace {
			// Adds `identifier` to the StringRefAttr-array attribute named
			// `attr_name` on `fn` if not already present. Used to track function-
			// level locals/names sets that the bytecode emitter consumes.
			void add_identifier_to(mlir::func::FuncOp fn,
				mlir::StringRef attr_name,
				mlir::StringRef identifier,
				mlir::OpBuilder &builder)
			{
				if (fn->hasAttr(attr_name)) {
					auto arr = mlir::cast<mlir::ArrayAttr>(fn->getAttr(attr_name)).getValue();
					if (std::find_if(arr.begin(),
							arr.end(),
							[identifier](mlir::Attribute attr) {
								return mlir::cast<mlir::StringAttr>(attr).getValue() == identifier;
							})
						!= arr.end()) {
						return;
					}
					std::vector<StringRef> names_vec;
					std::transform(arr.begin(),
						arr.end(),
						std::back_inserter(names_vec),
						[](mlir::Attribute attr) {
							return mlir::cast<mlir::StringAttr>(attr).getValue();
						});
					names_vec.emplace_back(identifier);
					fn->setAttr(attr_name, builder.getStrArrayAttr(names_vec));
				} else {
					fn->setAttr(attr_name, builder.getStrArrayAttr({ identifier }));
				}
			}

			void add_identifier(mlir::func::FuncOp fn,
				mlir::StringRef identifier,
				mlir::OpBuilder &builder)
			{
				add_identifier_to(fn, "names", identifier, builder);
			}
		}// namespace

		// LoadStore patterns moved to LoadStorePatterns.cpp.

		// CallFunctionLowering moved to FunctionPatterns.cpp.

		// Arith patterns (BinaryOp, InplaceOp, CompareOp, CastToBoolOp, the four
		// unary ops, and the py_kind_to_binary_op helper) moved to
		// ArithPatterns.cpp.
		// ConditionalBranch / CondBranchSubclass / LoadAssertionError / Raise /
		// WithExceptStart / ClearExceptionState / Yield / YieldFrom /
		// UnpackSequence / UnpackExpand / GetAwaitable moved to
		// ControlFlowPatterns.cpp.

		// Generic 1:1 lowering for ops whose source and target dialect schemas
		// match exactly: same operand types/order, same attribute names/types,
		// same result types. Forwards everything from source op to target op.
		template<typename From, typename To>
		struct DirectReplaceLowering : public mlir::OpRewritePattern<From>
		{
			using mlir::OpRewritePattern<From>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(From op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.template replaceOpWithNewOp<To>(op,
					op.getOperation()->getResultTypes(),
					op.getOperation()->getOperands(),
					op.getOperation()->getAttrs());
				return mlir::success();
			}
		};

		// Like DirectReplaceLowering, but additionally registers the name
		// returned by NameGetter in the parent FuncOp's "names" attribute.
		// Used for ops whose names must appear in the function-level name
		// table the bytecode emitter consumes (LoadMethod, StoreAttribute).
		template<typename From, typename To, auto NameGetter>
		struct DirectReplaceRegisterName : public mlir::OpRewritePattern<From>
		{
			using mlir::OpRewritePattern<From>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(From op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto parent_fn = op->template getParentOfType<mlir::func::FuncOp>();
				add_identifier(parent_fn, (op.*NameGetter)(), rewriter);
				rewriter.template replaceOpWithNewOp<To>(op,
					op.getOperation()->getResultTypes(),
					op.getOperation()->getOperands(),
					op.getOperation()->getAttrs());
				return mlir::success();
			}
		};

		// LoadAssertionErrorOpLowering moved to ControlFlowPatterns.cpp.

		// ReturnOpLowering moved to FunctionPatterns.cpp.

		// Collection patterns (BuildDict, BuildList, BuildTuple, BuildSet,
		// BuildString, FormatValue, ListAppend, DictAdd, SetAdd, BuildSlice,
		// build_list_with_expansion helper) moved to CollectionPatterns.cpp.
		// The attribute/subscript family below (LoadAttributeOpLowering etc.)
		// still lives here pending its own split.

		// Attribute / subscript patterns (LoadAttribute, DeleteAttribute,
		// LoadMethod, BinarySubscript, StoreSubscript, DeleteSubscript,
		// StoreAttribute) moved to AttributeSubscriptPatterns.cpp.
		// BuildSliceOpLowering moved to CollectionPatterns.cpp.


		struct ForLoopOpLowering : public mlir::OpRewritePattern<mlir::py::ForLoopOp>
		{
		  private:
			std::function<WalkResult(mlir::Operation *)> yield_op_callback(
				mlir::PatternRewriter &rewriter,
				mlir::Block *condition_start,
				mlir::Block *end_block) const
			{
				return [this, &rewriter, condition_start, end_block](mlir::Operation *operation) {
					auto parent_is_orelse = [](mlir::Operation *operation) {
						auto forloop_op = operation->getParentOfType<mlir::py::ForLoopOp>();
						if (!forloop_op) { return false; }
						return &forloop_op.getOrelse() == operation->getParentRegion();
					};
					// llvm::outs() << "ForOpLowering 1:\n";
					// operation->print(llvm::outs());
					// llvm::outs() << '\n';
					// llvm::outs().flush();

					if (auto loop = mlir::dyn_cast<mlir::py::ForLoopOp>(operation)) {
						if (loop.getOrelse().empty()) { return WalkResult::skip(); }
						// llvm::outs() << "ForOpLowering - ForLoopOp or else\n";
						// loop.getOrelse().front().print(llvm::outs());
						// llvm::outs() << '\n';
						// llvm::outs().flush();
						loop.getOrelse().walk<WalkOrder::PreOrder>(
							yield_op_callback(rewriter, condition_start, end_block));
						return WalkResult::skip();
					}
					if (auto loop = mlir::dyn_cast<mlir::py::WhileOp>(operation)) {
						if (loop.getOrelse().empty()) { return WalkResult::skip(); }
						// llvm::outs() << "ForOpLowering - WhileOp or else\n";
						loop.getOrelse().walk<WalkOrder::PreOrder>(
							yield_op_callback(rewriter, condition_start, end_block));
						return WalkResult::skip();
					}

					// llvm::outs() << "ForOpLowering 2:\n";
					// operation->print(llvm::outs());
					// llvm::outs() << '\n';
					// llvm::outs().flush();

					if (auto yield_op = mlir::dyn_cast<mlir::py::BranchYieldOp>(operation)) {
						static_assert(mlir::py::BranchYieldOp::hasTrait<mlir::OpTrait::
								HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerOp>::
									Impl>());
						if (!yield_op.getKind().has_value()
							&& mlir::isa<TryOp, WithOp, TryHandlerOp>(yield_op->getParentOp())) {
							return WalkResult::advance();
						}
						// is this hacky? maybe there is a better way of ignoring the else branch of
						// a for loop
						if (parent_is_orelse(operation)) { return WalkResult::advance(); }
						rewriter.setInsertionPoint(yield_op);
						if (!yield_op.getKind().has_value()
							|| yield_op.getKind().value() == py::LoopOpKind::continue_) {
							rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
								yield_op, condition_start);
						} else if (yield_op.getKind().value() == py::LoopOpKind::break_) {
							rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield_op, end_block);
						}
					}
					return WalkResult::advance();
				};
			}

			std::vector<mlir::Value> getIterators(mlir::py::ForLoopOp op,
				mlir::emitpybytecode::GetIter current_iterator) const
			{
				std::vector<mlir::Value> iterators;

				iterators.push_back(current_iterator);

				auto parent = op->getParentOfType<mlir::py::ForLoopOp>();
				while (parent) {
					auto iterable = parent.getIterable();
					ASSERT(!iterable.getUsers().empty());
					auto iterator = *iterable.getUsers().begin();
					ASSERT(mlir::isa<mlir::emitpybytecode::GetIter>(*iterator));
					iterators.insert(
						iterators.end() - 1, mlir::cast<mlir::emitpybytecode::GetIter>(*iterator));
					parent = parent->getParentOfType<mlir::py::ForLoopOp>();
				}

				return iterators;
			}

		  public:
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
				// iterator_next_block->addArgument(iterator.getType(), op.getStep().getLoc());
				rewriter.setInsertionPointToEnd(initBlock);
				const auto &iterators = getIterators(op, iterator);
				rewriter.create<mlir::cf::BranchOp>(op.getStep().getLoc(), iterator_next_block);

				rewriter.setInsertionPointToStart(iterator_next_block);

				rewriter.create<mlir::emitpybytecode::ForIter>(op.getStep().getLoc(),
					// iterator_next_block->getArgument(0),
					iterators.front(),
					&op.getStep().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());

				ASSERT(!op.getStep().empty());
				auto *iterator_exit_block = &op.getStep().back();
				ASSERT(iterator_exit_block->getTerminator());
				// iterator_exit_block->print(llvm::outs());
				// llvm::outs() << '\n';
				// llvm::outs().flush();
				ASSERT(mlir::isa<mlir::py::BranchYieldOp>(iterator_exit_block->getTerminator()));

				rewriter.setInsertionPointToEnd(iterator_exit_block);
				rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
					iterator_exit_block->getTerminator(), &op.getBody().front() /*, iterators*/);

				auto *for_iter_block = rewriter.createBlock(&op.getBody());
				// for (const auto &it : iterators) {
				// 	for_iter_block->addArgument(it.getType(), op.getStep().getLoc());
				// }
				rewriter.create<mlir::emitpybytecode::ForIter>(op.getStep().getLoc(),
					iterators.front(),
					&op.getStep().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());

				ASSERT(op.getStep().getArguments().size() == 1);
				rewriter.inlineRegionBefore(
					op.getStep(), *op->getParentRegion(), endBlock->getIterator());

				// for (const auto &it : iterators) {
				// 	op.getBody().addArgument(it.getType(), op.getStep().getLoc());
				// }

				op.getBody().walk<WalkOrder::PreOrder>(
					yield_op_callback(rewriter, for_iter_block, endBlock));

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

				// llvm::outs() << "ForLoopOp rewrite end\n";
				// llvm::outs().flush();

				return success();
			}
		};

		struct WhileOpLowering : public mlir::OpRewritePattern<mlir::py::WhileOp>
		{
		  private:
			std::function<WalkResult(mlir::Operation *)> yield_op_callback(
				mlir::PatternRewriter &rewriter,
				mlir::Block *condition_start,
				mlir::Block *end_block) const
			{
				return [this, &rewriter, condition_start, end_block](mlir::Operation *operation) {
					// llvm::outs() << "WhileOpLowering 1:\n";
					// operation->print(llvm::outs());
					// llvm::outs() << '\n';
					// llvm::outs().flush();
					if (auto loop = mlir::dyn_cast<mlir::py::ForLoopOp>(operation)) {
						if (loop.getOrelse().empty()) { return WalkResult::skip(); }
						// llvm::outs() << "WhileOpLowering - ForLoopOp or else\n";
						loop.getOrelse().walk<WalkOrder::PreOrder>(
							yield_op_callback(rewriter, condition_start, end_block));
						return WalkResult::skip();
					}
					if (auto loop = mlir::dyn_cast<mlir::py::WhileOp>(operation)) {
						if (loop.getOrelse().empty()) { return WalkResult::skip(); }
						// llvm::outs() << "WhileOpLowering - WhileOp or else\n";
						loop.getOrelse().walk<WalkOrder::PreOrder>(
							yield_op_callback(rewriter, condition_start, end_block));
						return WalkResult::skip();
					}
					// llvm::outs() << "WhileOpLowering 2:\n";
					// operation->print(llvm::outs());
					// llvm::outs() << '\n';
					// llvm::outs().flush();
					if (auto yield_op = mlir::dyn_cast<mlir::py::BranchYieldOp>(operation)) {
						static_assert(mlir::py::BranchYieldOp::hasTrait<mlir::OpTrait::
								HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerOp>::
									Impl>());
						if (!yield_op.getKind().has_value()
							&& mlir::isa<TryOp, WithOp, TryHandlerOp>(yield_op->getParentOp())) {
							return WalkResult::advance();
						}
						rewriter.setInsertionPoint(yield_op);
						if (!yield_op.getKind().has_value()
							|| yield_op.getKind().value() == py::LoopOpKind::continue_) {
							rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
								yield_op, condition_start);
						} else if (yield_op.getKind().value() == py::LoopOpKind::break_) {
							rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield_op, end_block);
						}
					}
					return WalkResult::advance();
				};
			}

		  public:
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

				op.getBody().walk<WalkOrder::PreOrder>(
					yield_op_callback(rewriter, &condition_start, endBlock));

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

					// TODO: handle multiple handlers
					rewriter.setInsertionPointToStart(continue_block);
					rewriter.create<mlir::emitpybytecode::ClearExceptionState>(item.getLoc());
					rewriter.create<mlir::cf::BranchOp>(op.getLoc(), endBlock);
				}
				// rewriter.create<mlir::cf::BranchOp>(op.getLoc(), endBlock);

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
		ConversionTarget target(getContext());
		target.addLegalDialect<emitpybytecode::EmitPythonBytecodeDialect, mlir::BuiltinDialect>();

		target.addLegalOp<mlir::cf::BranchOp>();
		target.addLegalOp<mlir::func::ReturnOp>();
		target.addDynamicallyLegalOp<mlir::func::FuncOp>([](mlir::func::FuncOp op) {
			// don't convert this special function, which is the entry point of a module
			return op.isPrivate() && op.getSymName() == "__hidden_init__";
		});
		target.addIllegalDialect<PythonDialect>();

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

		// Currently ignoring the return value as it seems to always fail, even though the
		// transformation seems to generate the expected output
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