#include "Conversion/PythonToPythonBytecode/LoweringHelpers.hpp"
#include "Conversion/PythonToPythonBytecode/PatternPopulators.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"

#include "executable/bytecode/instructions/BinaryOperation.hpp"
#include "executable/bytecode/instructions/Unary.hpp"
#include "utilities.hpp"

#include "mlir/IR/PatternMatch.h"

namespace mlir::py {
namespace {

	// Translate py.{binary,inplace_op}'s ArithOpKind enum to the
	// bytecode-level BinaryOperation::Operation enum. The two are
	// deliberately decoupled: the dialect enum is part of the IR
	// contract; the bytecode enum is the wire format consumed by the
	// VM, so the mapping between the two must stay explicit.
	BinaryOperation::Operation py_kind_to_binary_op(mlir::py::ArithOpKind kind)
	{
		switch (kind) {
		case mlir::py::ArithOpKind::add:
			return BinaryOperation::Operation::PLUS;
		case mlir::py::ArithOpKind::sub:
			return BinaryOperation::Operation::MINUS;
		case mlir::py::ArithOpKind::mod:
			return BinaryOperation::Operation::MODULO;
		case mlir::py::ArithOpKind::mul:
			return BinaryOperation::Operation::MULTIPLY;
		case mlir::py::ArithOpKind::exp:
			return BinaryOperation::Operation::EXP;
		case mlir::py::ArithOpKind::div:
			return BinaryOperation::Operation::SLASH;
		case mlir::py::ArithOpKind::fldiv:
			return BinaryOperation::Operation::FLOORDIV;
		case mlir::py::ArithOpKind::mmul:
			return BinaryOperation::Operation::MATMUL;
		case mlir::py::ArithOpKind::lshift:
			return BinaryOperation::Operation::LEFTSHIFT;
		case mlir::py::ArithOpKind::rshift:
			return BinaryOperation::Operation::RIGHTSHIFT;
		case mlir::py::ArithOpKind::and_:
			return BinaryOperation::Operation::AND;
		case mlir::py::ArithOpKind::or_:
			return BinaryOperation::Operation::OR;
		case mlir::py::ArithOpKind::xor_:
			return BinaryOperation::Operation::XOR;
		}
		ASSERT_NOT_REACHED();
	}

	struct InplaceOpLowering : public mlir::OpRewritePattern<InplaceOp>
	{
		using OpRewritePattern<InplaceOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(InplaceOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto op_type = mlir::IntegerAttr::get(rewriter.getIntegerType(8, false),
				static_cast<uint8_t>(py_kind_to_binary_op(op.getKind())));

			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::InplaceOp>(
				op, op.getResult().getType(), op.getDst(), op.getSrc(), op_type);

			return success();
		}
	};

	struct BinaryOpLowering : public mlir::OpRewritePattern<mlir::py::BinaryOp>
	{
		using OpRewritePattern<mlir::py::BinaryOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::BinaryOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto op_type = mlir::IntegerAttr::get(rewriter.getIntegerType(8, false),
				static_cast<uint8_t>(py_kind_to_binary_op(op.getKind())));
			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BinaryOp>(
				op, op.getOutput().getType(), op.getLhs(), op.getRhs(), op_type);
			return success();
		}
	};

	// Trivial 1:1 lowering of a py.unary_* op to emitpybytecode.UNARY_OP
	// with the corresponding Unary::Operation enum baked in.
	template<typename From, Unary::Operation Kind>
	struct UnaryOpLowering : public mlir::OpRewritePattern<From>
	{
		using mlir::OpRewritePattern<From>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(From op, mlir::PatternRewriter &rewriter) const final
		{
			rewriter.template replaceOpWithNewOp<mlir::emitpybytecode::UnaryOp>(
				op, op.getOutput().getType(), op.getInput(), static_cast<uint8_t>(Kind));
			return mlir::success();
		}
	};

	using PositiveOpLowering = UnaryOpLowering<mlir::py::PositiveOp, Unary::Operation::POSITIVE>;
	using NegativeOpLowering = UnaryOpLowering<mlir::py::NegativeOp, Unary::Operation::NEGATIVE>;
	using InvertOpLowering = UnaryOpLowering<mlir::py::InvertOp, Unary::Operation::INVERT>;
	using NotOpLowering = UnaryOpLowering<mlir::py::NotOp, Unary::Operation::NOT>;

	struct CompareOpLowering : public mlir::OpRewritePattern<mlir::py::CompareOp>
	{
		using OpRewritePattern<mlir::py::CompareOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::CompareOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto lhs = op.getLhs();
			auto rhs = op.getRhs();
			auto op_type = mlir::IntegerAttr::get(
				rewriter.getIntegerType(8, false), static_cast<uint8_t>(op.getPredicate()));

			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::Compare>(
				op, op.getOutput().getType(), lhs, rhs, op_type);

			return success();
		}
	};

	struct CastToBoolOpLowering : public mlir::OpRewritePattern<mlir::py::CastToBoolOp>
	{
		using OpRewritePattern<mlir::py::CastToBoolOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::CastToBoolOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::CastToBool>(
				op, op.getValue().getType(), op.getValue());
			return success();
		}
	};

}// namespace

void populateArithPatterns(mlir::RewritePatternSet &patterns)
{
	auto *ctx = patterns.getContext();
	patterns.add<BinaryOpLowering, InplaceOpLowering, CompareOpLowering, CastToBoolOpLowering>(ctx);
	patterns.add<PositiveOpLowering, NegativeOpLowering, InvertOpLowering, NotOpLowering>(ctx);
}

}// namespace mlir::py
