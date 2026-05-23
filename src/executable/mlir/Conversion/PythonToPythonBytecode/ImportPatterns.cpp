#include "Conversion/PythonToPythonBytecode/LoweringHelpers.hpp"
#include "Conversion/PythonToPythonBytecode/PatternPopulators.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"
#include "Dialect/Python/IR/PythonTypes.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::py {
namespace {

	// py.import lowers to LOAD_CONST(level) + a BuildTuple over the
	// from_list constants, then IMPORT_NAME. The bytecode form takes
	// the level and from-list as actual operands, so the conversion
	// materialises them as constants here rather than threading them
	// through as attributes on the bytecode op.
	struct ImportOpLowering : public mlir::OpRewritePattern<mlir::py::ImportOp>
	{
		using OpRewritePattern<mlir::py::ImportOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::ImportOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto name = op.getName();
			auto level = rewriter.create<mlir::emitpybytecode::ConstantOp>(
				op.getLoc(), op.getModule().getType(), rewriter.getUI32IntegerAttr(op.getLevel()));
			std::vector<mlir::Value> els;
			for (auto attr : op.getFromList()) {
				auto from = mlir::cast<mlir::StringAttr>(attr);
				els.push_back(rewriter.create<mlir::emitpybytecode::ConstantOp>(
					op.getLoc(), op.getModule().getType(), from));
			}
			auto from_list = rewriter.create<mlir::emitpybytecode::BuildTuple>(
				op.getLoc(), op.getModule().getType(), els);
			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ImportName>(
				op, op.getModule().getType(), name, level, from_list);

			return success();
		}
	};

	using ImportAllOpLowering =
		detail::DirectReplaceLowering<mlir::py::ImportAllOp, mlir::emitpybytecode::ImportAll>;

	using ImportFromOpLowering =
		detail::DirectReplaceLowering<mlir::py::ImportFromOp, mlir::emitpybytecode::ImportFrom>;

}// namespace

void populateImportPatterns(mlir::RewritePatternSet &patterns)
{
	patterns.add<ImportOpLowering, ImportFromOpLowering, ImportAllOpLowering>(
		patterns.getContext());
}

}// namespace mlir::py
