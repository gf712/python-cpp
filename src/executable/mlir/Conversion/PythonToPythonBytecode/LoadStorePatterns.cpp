#include "Conversion/PythonToPythonBytecode/LoweringHelpers.hpp"
#include "Conversion/PythonToPythonBytecode/PatternPopulators.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/PythonAttributes.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"

#include "mlir/IR/PatternMatch.h"

namespace mlir::py {
namespace {

	// py.constant -> emitpybytecode.LOAD_CONST, with a special case for
	// the ellipsis attribute (which has its own bytecode-level op).
	struct ConstantLoadLowering : public mlir::OpRewritePattern<py::ConstantOp>
	{
		using OpRewritePattern<py::ConstantOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(py::ConstantOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto constant_value = op.getValue();

			auto ellipsis =
				mlir::detail::AttributeUniquer::get<mlir::py::EllipsisAttr>(getContext());
			if (op.getValue() == ellipsis) {
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadEllipsisOp>(
					op, op.getOutput().getType());
				return success();
			}

			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ConstantOp>(
				op, op.getOutput().getType(), constant_value);

			return success();
		}
	};

	// Trivial 1:1 lowering of py.load_* (and similar single-result name-
	// referencing ops) to the corresponding emitpybytecode.LOAD_* op.
	template<typename From, typename To, detail::NameKind Kind = detail::NameKind::None>
	struct LoadNameLikeLowering : public mlir::OpRewritePattern<From>
	{
		using mlir::OpRewritePattern<From>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(From op, mlir::PatternRewriter &rewriter) const final
		{
			detail::register_name_with_parent(op, op.getNameAttr(), rewriter, Kind);
			rewriter.template replaceOpWithNewOp<To>(
				op, op.getOutput().getType(), op.getNameAttr());
			return mlir::success();
		}
	};

	// Trivial 1:1 lowering of py.store_* to emitpybytecode.STORE_*.
	template<typename From, typename To, detail::NameKind Kind = detail::NameKind::None>
	struct StoreNameLikeLowering : public mlir::OpRewritePattern<From>
	{
		using mlir::OpRewritePattern<From>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(From op, mlir::PatternRewriter &rewriter) const final
		{
			detail::register_name_with_parent(op, op.getNameAttr(), rewriter, Kind);
			rewriter.template replaceOpWithNewOp<To>(op, op.getNameAttr(), op.getValue());
			return mlir::success();
		}
	};

	// Trivial 1:1 lowering of py.delete_* to emitpybytecode.DELETE_*.
	template<typename From, typename To, detail::NameKind Kind = detail::NameKind::None>
	struct DeleteNameLikeLowering : public mlir::OpRewritePattern<From>
	{
		using mlir::OpRewritePattern<From>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(From op, mlir::PatternRewriter &rewriter) const final
		{
			detail::register_name_with_parent(op, op.getNameAttr(), rewriter, Kind);
			rewriter.template replaceOpWithNewOp<To>(op, op.getNameAttr());
			return mlir::success();
		}
	};

	using LoadNameLowering = LoadNameLikeLowering<py::LoadNameOp, mlir::emitpybytecode::LoadNameOp>;
	using LoadDerefLowering =
		LoadNameLikeLowering<py::LoadDerefOp, mlir::emitpybytecode::LoadDerefOp>;
	using LoadFastLowering = LoadNameLikeLowering<py::LoadFastOp,
		mlir::emitpybytecode::LoadFastOp,
		detail::NameKind::Local>;
	using LoadGlobalLowering = LoadNameLikeLowering<py::LoadGlobalOp,
		mlir::emitpybytecode::LoadGlobalOp,
		detail::NameKind::Global>;

	using StoreNameLowering =
		StoreNameLikeLowering<py::StoreNameOp, mlir::emitpybytecode::StoreNameOp>;
	using StoreDerefLowering =
		StoreNameLikeLowering<py::StoreDerefOp, mlir::emitpybytecode::StoreDerefOp>;
	using StoreFastLowering = StoreNameLikeLowering<py::StoreFastOp,
		mlir::emitpybytecode::StoreFastOp,
		detail::NameKind::Local>;
	using StoreGlobalLowering = StoreNameLikeLowering<py::StoreGlobalOp,
		mlir::emitpybytecode::StoreGlobalOp,
		detail::NameKind::Global>;

	using DeleteNameLowering =
		DeleteNameLikeLowering<py::DeleteNameOp, mlir::emitpybytecode::DeleteNameOp>;
	using DeleteDerefLowering =
		DeleteNameLikeLowering<py::DeleteDerefOp, mlir::emitpybytecode::DeleteDerefOp>;
	using DeleteFastLowering = DeleteNameLikeLowering<py::DeleteFastOp,
		mlir::emitpybytecode::DeleteFastOp,
		detail::NameKind::Local>;
	using DeleteGlobalLowering = DeleteNameLikeLowering<py::DeleteGlobalOp,
		mlir::emitpybytecode::DeleteGlobalOp,
		detail::NameKind::Global>;

}// namespace

void populateLoadStorePatterns(mlir::RewritePatternSet &patterns)
{
	auto *ctx = patterns.getContext();
	patterns.add<ConstantLoadLowering,
		LoadFastLowering,
		LoadNameLowering,
		LoadGlobalLowering,
		LoadDerefLowering>(ctx);
	patterns.add<StoreFastLowering, StoreGlobalLowering, StoreNameLowering, StoreDerefLowering>(
		ctx);
	patterns.add<DeleteFastLowering, DeleteGlobalLowering, DeleteNameLowering, DeleteDerefLowering>(
		ctx);
}

}// namespace mlir::py
