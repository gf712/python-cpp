#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <algorithm>
#include <vector>

namespace mlir::py::detail {

// Adds `identifier` to the StrArrayAttr named `attr_name` on `fn` if
// not already present. Used by the conversion patterns to track the
// function-level "locals" / "names" sets that the bytecode emitter
// consumes.
inline void add_identifier_to(mlir::func::FuncOp fn,
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
		std::vector<mlir::StringRef> names_vec;
		std::transform(
			arr.begin(), arr.end(), std::back_inserter(names_vec), [](mlir::Attribute attr) {
				return mlir::cast<mlir::StringAttr>(attr).getValue();
			});
		names_vec.emplace_back(identifier);
		fn->setAttr(attr_name, builder.getStrArrayAttr(names_vec));
	} else {
		fn->setAttr(attr_name, builder.getStrArrayAttr({ identifier }));
	}
}

inline void
	add_identifier(mlir::func::FuncOp fn, mlir::StringRef identifier, mlir::OpBuilder &builder)
{
	add_identifier_to(fn, "names", identifier, builder);
}

// Categorizes how a Load/Store/Delete pattern should register its name
// with the parent FuncOp before lowering.
enum class NameKind {
	None,// no registration (local-scope name / cell deref)
	Local,// registers in the func's "locals" attribute
	Global,// registers in the func's "names" attribute
};

inline void register_name_with_parent(mlir::Operation *op,
	mlir::StringAttr name,
	mlir::OpBuilder &builder,
	NameKind kind)
{
	if (kind == NameKind::None) { return; }
	auto fn = mlir::cast_or_null<mlir::func::FuncOp>(op->getParentOp());
	assert(fn);
	add_identifier_to(fn, kind == NameKind::Local ? "locals" : "names", name.getValue(), builder);
}

// Generic 1:1 lowering for ops whose source and target dialect schemas
// match exactly: same operand types/order, same attribute names/types,
// same result types. Forwards everything from source op to target op.
template<typename From, typename To>
struct DirectReplaceLowering : public mlir::OpRewritePattern<From>
{
	using mlir::OpRewritePattern<From>::OpRewritePattern;

	mlir::LogicalResult matchAndRewrite(From op, mlir::PatternRewriter &rewriter) const final
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

	mlir::LogicalResult matchAndRewrite(From op, mlir::PatternRewriter &rewriter) const final
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

}// namespace mlir::py::detail
