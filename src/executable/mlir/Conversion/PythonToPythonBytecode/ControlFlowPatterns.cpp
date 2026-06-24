#include "Conversion/PythonToPythonBytecode/LoweringHelpers.hpp"
#include "Conversion/PythonToPythonBytecode/PatternPopulators.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"

#include "utilities.hpp"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::py {
namespace {

	struct ConditionalBranchOpLowering : public mlir::OpRewritePattern<mlir::cf::CondBranchOp>
	{
		using OpRewritePattern<mlir::cf::CondBranchOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::cf::CondBranchOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto cond = (*op.getODSOperands(0).begin());
			ASSERT(cond.getDefiningOp());
			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::JumpIfFalse>(op,
				cond,
				op.getTrueDest(),
				op.getTrueDestOperands(),
				op.getFalseDest(),
				op.getFalseDestOperands());
			return success();
		}
	};

	struct CondBranchSubclassOpLowering
		: public mlir::OpRewritePattern<mlir::py::CondBranchSubclassOp>
	{
		using OpRewritePattern<mlir::py::CondBranchSubclassOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::CondBranchSubclassOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::JumpIfNotException>(op,
				op.getObjectType(),
				op.getTrueDestOperands(),
				op.getFalseDestOperands(),
				op.getTrueDest(),
				op.getFalseDest());

			return success();
		}
	};

	using LoadAssertionErrorOpLowering = detail::DirectReplaceLowering<mlir::py::LoadAssertionError,
		mlir::emitpybytecode::LoadAssertionError>;

	using LoadExceptionOpLowering =
		detail::DirectReplaceLowering<mlir::py::LoadException, mlir::emitpybytecode::LoadException>;

	using WithExceptStartOpLowering = detail::DirectReplaceLowering<mlir::py::WithExceptStartOp,
		mlir::emitpybytecode::WithExceptStart>;

	using ClearExceptionStateOpLowering =
		detail::DirectReplaceLowering<mlir::py::ClearExceptionStateOp,
			mlir::emitpybytecode::ClearExceptionState>;

	struct RaiseOpLowering : public mlir::OpRewritePattern<mlir::py::RaiseOp>
	{
		using OpRewritePattern<mlir::py::RaiseOp>::OpRewritePattern;

		/// Find the first parent operation of the given type, or nullptr if there is
		/// no ancestor operation.
		template<typename... ParentTs> static mlir::Operation *getParentOfType(mlir::Region *region)
		{
			do {
				if ((... || mlir::isa<ParentTs>(*region->getParentOp())))
					return region->getParentOp();
			} while ((region = region->getParentRegion()));
			return nullptr;
		}

		static mlir::Block *get_handler(mlir::Operation *op, mlir::PatternRewriter &rewriter)
		{
			// find possible catch block in order to not clobber an active result register
			auto *handler_op =
				getParentOfType<mlir::py::TryOp, mlir::py::WithOp, mlir::func::FuncOp>(
					op->getParentRegion());
			ASSERT(handler_op);
			return llvm::TypeSwitch<mlir::Operation *, mlir::Block *>(handler_op)
				.Case([](mlir::py::TryOp op) {
					return op.getHandlers().empty() ? &op.getFinally().front()
													: &op.getHandlers().front().front();
				})
				.Case([](mlir::py::WithOp op) { return op->getParentOp()->getBlock(); })
				.Case([&rewriter](mlir::func::FuncOp op) {
					auto insertion_point = rewriter.getInsertionPoint();
					auto *return_block = rewriter.createBlock(&op.getRegion());
					auto value =
						rewriter.create<mlir::py::ConstantOp>(op.getLoc(), rewriter.getNoneType());
					rewriter.create<mlir::func::ReturnOp>(op.getLoc(), mlir::ValueRange{ value });
					rewriter.setInsertionPoint(insertion_point->getBlock(), insertion_point);
					return return_block;
				})
				.Default([](mlir::Operation *) -> mlir::Block * {
					// Structurally unreachable: getParentOfType only
					// walks for TryOp/WithOp/FuncOp, and the preceding
					// ASSERT rules out nullptr, so one of the three
					// Cases above must match.
					ASSERT_NOT_REACHED();
				});
		}

		mlir::LogicalResult matchAndRewrite(mlir::py::RaiseOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			if (auto exception = op.getException()) {
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::RaiseVarargs>(
					op, exception, op.getCause(), get_handler(op, rewriter));
			} else {
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ReRaiseOp>(
					op, get_handler(op, rewriter));
			}

			return success();
		}
	};

	using YieldOpLowering =
		detail::DirectReplaceLowering<mlir::py::YieldOp, mlir::emitpybytecode::Yield>;

	struct YieldFromOpLowering : public mlir::OpRewritePattern<mlir::py::YieldFromOp>
	{
		using OpRewritePattern<mlir::py::YieldFromOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::YieldFromOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto iterator = rewriter.create<mlir::emitpybytecode::YieldFromIter>(
				op.getLoc(), op.getIterable().getType(), op.getIterable());
			auto value = rewriter.create<mlir::py::ConstantOp>(op.getLoc(), rewriter.getNoneType());

			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::YieldFrom>(
				op, iterator.getType(), iterator, value);

			return success();
		}
	};

	using UnpackSequenceOpLowering = detail::DirectReplaceLowering<mlir::py::UnpackSequenceOp,
		mlir::emitpybytecode::UnpackSequenceOp>;

	using UnpackExpandOpLowering = detail::DirectReplaceLowering<mlir::py::UnpackExpandOp,
		mlir::emitpybytecode::UnpackExpandOp>;

	using GetAwaitableOpLowering = detail::DirectReplaceLowering<mlir::py::GetAwaitableOp,
		mlir::emitpybytecode::GetAwaitableOp>;

}// namespace

void populateControlFlowPatterns(mlir::RewritePatternSet &patterns)
{
	auto *ctx = patterns.getContext();
	patterns.add<ConditionalBranchOpLowering,
		CondBranchSubclassOpLowering,
		LoadAssertionErrorOpLowering,
		LoadExceptionOpLowering,
		RaiseOpLowering,
		WithExceptStartOpLowering,
		ClearExceptionStateOpLowering,
		YieldOpLowering,
		YieldFromOpLowering,
		UnpackSequenceOpLowering,
		UnpackExpandOpLowering,
		GetAwaitableOpLowering>(ctx);
}

}// namespace mlir::py
