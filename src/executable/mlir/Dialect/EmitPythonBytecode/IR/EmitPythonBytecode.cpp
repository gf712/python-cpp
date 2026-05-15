#include "EmitPythonBytecode/IR/EmitPythonBytecode.hpp"

#include "Python/IR/Dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "EmitPythonBytecode/IR/EmitPythonBytecodeDialect.cpp.inc"

namespace mlir {
namespace emitpybytecode {
	void EmitPythonBytecodeDialect::initialize()
	{
		addOperations<
#define GET_OP_LIST
#include "EmitPythonBytecode/IR/EmitPythonBytecodeOps.cpp.inc"
			>();
	}

	SuccessorOperands JumpIfFalse::getSuccessorOperands(unsigned index)
	{
		assert(index < getNumSuccessors() && "invalid successor index");
		return SuccessorOperands(
			index == 0 ? getTrueDestOperandsMutable() : getFalseDestOperandsMutable());
	}

	SuccessorOperands JumpIfNotException::getSuccessorOperands(unsigned index)
	{
		assert(index < getNumSuccessors() && "invalid successor index");
		return SuccessorOperands(
			index == 0 ? getTrueDestOperandsMutable() : getFalseDestOperandsMutable());
	}

	SuccessorOperands ForIter::getSuccessorOperands(unsigned index)
	{
		if (index == 0) {
			return SuccessorOperands(1, mlir::MutableOperandRange{ getOperation(), 0, 0 });
		}
		return SuccessorOperands(0, mlir::MutableOperandRange{ getOperation(), 0, 0 });
	}

	namespace {
		// Register-pressure relief for large dict literals. A single
		// BUILD_DICT consuming N keys + N values forces 2N live values
		// to coexist in registers immediately before the call; the
		// linear-scan allocator handles that by spilling, which is
		// expensive (no stack slots — only register-register moves +
		// Push/Pop). For literals with more than ~10 entries, splitting
		// into an empty BUILD_DICT followed by streamed DICT_ADD ops
		// (each emitted next to its value's defining op) drops the
		// live-value count to 3 (dict, key, value). The threshold is
		// empirical; a future register-pressure-aware allocation pass
		// (plan step 19) would let us pick this dynamically.
		struct ExpandLargeBuildDict : public mlir::OpRewritePattern<BuildDict>
		{
			using mlir::OpRewritePattern<BuildDict>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(BuildDict op,
				mlir::PatternRewriter &rewriter) const final
			{
				if (op.getValues().size() <= 10) { return mlir::failure(); }
				auto keys = op.getKeys();
				auto values = op.getValues();
				rewriter.setInsertionPointAfterValue(keys.front());
				auto result = rewriter.create<BuildDict>(
					op->getLoc(), op.getOutput().getType(), mlir::ValueRange{}, mlir::ValueRange{});

				for (auto [key, value] : llvm::zip(keys, values)) {
					rewriter.setInsertionPointAfterValue(value);
					rewriter.create<DictAdd>(op.getLoc(), result, key, value);
				}
				rewriter.replaceOp(op, result);
				return mlir::success();
			}
		};
	}// namespace

	void BuildDict::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
		mlir::MLIRContext *context)
	{
		patterns.add<ExpandLargeBuildDict>(context);
	}

	namespace {
		// Register-pressure relief for all-constants list literals
		// (`[1, 2, "x"]`). A direct BUILD_LIST with N operands holds N
		// live values until the op runs; rewriting to an empty
		// BUILD_LIST + a single LOAD_CONST tuple + LIST_EXTEND keeps the
		// live-value count at 2 (list, tuple). Same motivation and same
		// "step 19 will subsume this" caveat as ExpandLargeBuildDict.
		struct FoldAllConstBuildListIntoExtend : public mlir::OpRewritePattern<BuildList>
		{
			using mlir::OpRewritePattern<BuildList>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(BuildList op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto elements_ops = op.getElements();
				if (elements_ops.empty()) { return mlir::failure(); }
				std::vector<mlir::Attribute> elements;
				elements.reserve(elements_ops.size());
				for (auto el : elements_ops) {
					auto k = el.getDefiningOp<ConstantOp>();
					if (!k) { return mlir::failure(); }
					elements.push_back(k.getValue());
				}
				auto loc = op.getLoc();
				auto output_type = op.getOutput().getType();
				auto list = rewriter.create<BuildList>(loc, output_type, mlir::ValueRange{});
				auto tuple = rewriter.create<ConstantOp>(
					loc, output_type, mlir::ArrayAttr::get(getContext(), elements));
				rewriter.create<ListExtend>(loc, list, tuple);
				rewriter.replaceOp(op, list);
				return mlir::success();
			}
		};
	}// namespace

	void BuildList::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
		mlir::MLIRContext *context)
	{
		patterns.add<FoldAllConstBuildListIntoExtend>(context);
	}

	mlir::LogicalResult ConstantOp::verify()
	{
		mlir::Attribute attr = getValue();
		// Accepted attribute kinds match the TypeSwitch in
		// PythonBytecodeEmitter::emitOperation(emitpybytecode::ConstantOp).
		// EllipsisAttr is not in the list - the conversion pass lowers it to
		// LoadEllipsisOp instead.
		if (mlir::isa<mlir::FloatAttr,
				mlir::BoolAttr,
				mlir::UnitAttr,
				mlir::StringAttr,
				mlir::IntegerAttr,
				mlir::DenseIntElementsAttr,
				mlir::ArrayAttr>(attr)) {
			return mlir::success();
		}
		return emitOpError() << "value attribute has unsupported kind: " << attr;
	}
}// namespace emitpybytecode
}// namespace mlir

#define GET_OP_CLASSES
#include "EmitPythonBytecode/IR/EmitPythonBytecodeOps.cpp.inc"

// #define GET_TYPEDEF_CLASSES
// #include "EmitPythonBytecode/IR/EmitPythonBytecodeTypes.cpp.inc"
