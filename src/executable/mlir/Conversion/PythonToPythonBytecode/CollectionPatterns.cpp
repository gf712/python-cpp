#include "Conversion/PythonToPythonBytecode/LoweringHelpers.hpp"
#include "Conversion/PythonToPythonBytecode/PatternPopulators.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"

#include "utilities.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <vector>

namespace mlir::py {
namespace {

	struct BuildDictOpLowering : public mlir::OpRewritePattern<mlir::py::BuildDictOp>
	{
		using OpRewritePattern<mlir::py::BuildDictOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::BuildDictOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			const auto &requires_expansion = op.getRequiresExpansion();
			if (std::any_of(requires_expansion.begin(),
					requires_expansion.end(),
					[](const auto &el) { return el; })) {
				std::optional<mlir::Value> result;
				std::vector<mlir::Value> keys;
				std::vector<mlir::Value> values;

				for (auto [key, value, to_expand] :
					llvm::zip(op.getKeys(), op.getValues(), op.getRequiresExpansion())) {
					if (to_expand) {
						if (!result.has_value()) {
							result = rewriter.create<mlir::emitpybytecode::BuildDict>(
								op.getLoc(), op.getOutput().getType(), keys, values);
							keys.clear();
							values.clear();
						}
						rewriter.create<mlir::emitpybytecode::DictUpdate>(
							op.getLoc(), *result, value);
					} else {
						if (!result.has_value()) {
							keys.push_back(key);
							values.push_back(value);
						} else {
							ASSERT(keys.empty());
							ASSERT(values.empty());
							rewriter.create<mlir::emitpybytecode::DictAdd>(
								op.getLoc(), *result, key, value);
						}
					}
				}

				ASSERT(result.has_value());
				ASSERT(keys.empty());
				ASSERT(values.empty());

				rewriter.replaceOp(op, { *result });
			} else {
				// Plain dict literal: lower 1:1. The register-pressure
				// optimisation for large literals lives as a
				// canonicalize pattern on emitpybytecode.BuildDict
				// (ExpandLargeBuildDict in EmitPythonBytecode.cpp) so
				// any path that reaches that op benefits, not just the
				// one through this lowering.
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildDict>(
					op, op.getOutput().getType(), op.getKeys(), op.getValues());
			}

			return success();
		}
	};

	// Build a list incrementally by walking (element, requires_expansion)
	// pairs: each "expand" entry produces a ListExtend (unpacking *args
	// or **kwargs in literals), each non-expand entry produces a
	// ListAppend. Returns the resulting BuildList value, which both
	// BuildListOp and BuildTupleOp lowerings hand off to their final
	// step (replaceOp / wrap in ListToTuple respectively).
	mlir::emitpybytecode::BuildList build_list_with_expansion(mlir::PatternRewriter &rewriter,
		mlir::Location loc,
		mlir::Type list_type,
		mlir::ValueRange elements,
		llvm::ArrayRef<bool> requires_expansion)
	{
		auto list =
			rewriter.create<mlir::emitpybytecode::BuildList>(loc, list_type, mlir::ValueRange{});
		for (auto [el, expand] : llvm::zip(elements, requires_expansion)) {
			if (expand) {
				rewriter.create<mlir::emitpybytecode::ListExtend>(loc, list, el);
			} else {
				rewriter.create<mlir::emitpybytecode::ListAppend>(loc, list, el);
			}
		}
		return list;
	}

	struct BuildListOpLowering : public mlir::OpRewritePattern<mlir::py::BuildListOp>
	{
		using OpRewritePattern<mlir::py::BuildListOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::BuildListOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			const auto &requires_expansion = op.getRequiresExpansion();
			if (std::any_of(requires_expansion.begin(),
					requires_expansion.end(),
					[](const auto &el) { return el == 1; })) {
				auto list = build_list_with_expansion(rewriter,
					op.getLoc(),
					op.getOutput().getType(),
					op.getElements(),
					requires_expansion);
				rewriter.replaceOp(op, list);
			} else {
				// Plain list literal: lower 1:1. The all-constants
				// register-pressure rewrite lives as a canonicalize
				// pattern on emitpybytecode.BuildList
				// (FoldAllConstBuildListIntoExtend in
				// EmitPythonBytecode.cpp), so any caller landing on
				// the bytecode-level op benefits, not just this path.
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildList>(
					op, op.getOutput().getType(), op.getElements());
			}

			return success();
		}
	};

	using ListAppendOpLowering =
		detail::DirectReplaceLowering<mlir::py::ListAppendOp, mlir::emitpybytecode::ListAppend>;

	using DictAddOpLowering =
		detail::DirectReplaceLowering<mlir::py::DictAddOp, mlir::emitpybytecode::DictAdd>;

	struct BuildTupleOpLowering : public mlir::OpRewritePattern<mlir::py::BuildTupleOp>
	{
		using OpRewritePattern<mlir::py::BuildTupleOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::BuildTupleOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			const auto &requires_expansion = op.getRequiresExpansion();
			if (std::any_of(requires_expansion.begin(),
					requires_expansion.end(),
					[](const auto &el) { return el == 1; })) {
				auto list = build_list_with_expansion(rewriter,
					op.getLoc(),
					op.getOutput().getType(),
					op.getElements(),
					requires_expansion);
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ListToTuple>(
					op, op.getOutput().getType(), list);
			} else {
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildTuple>(
					op, op.getOutput().getType(), op.getElements());
			}

			return success();
		}
	};

	struct BuildSetOpLowering : public mlir::OpRewritePattern<mlir::py::BuildSetOp>
	{
		using OpRewritePattern<mlir::py::BuildSetOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::BuildSetOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			const auto &requires_expansion = op.getRequiresExpansion();
			if (std::any_of(requires_expansion.begin(),
					requires_expansion.end(),
					[](const auto &el) { return el == 1; })) {
				std::vector<mlir::Value> elements;
				std::optional<mlir::Value> set;
				for (auto [el, expand] : llvm::zip(op.getElements(), requires_expansion)) {
					if (expand) {
						if (!set.has_value()) {
							set = rewriter.create<mlir::emitpybytecode::BuildSet>(
								op->getLoc(), op.getOutput().getType(), elements);
						} else {
							for (auto el : elements) {
								rewriter.create<mlir::emitpybytecode::SetAdd>(
									op.getLoc(), *set, el);
							}
						}
						elements.clear();
						rewriter.create<mlir::emitpybytecode::SetUpdate>(op.getLoc(), *set, el);
					} else {
						elements.push_back(el);
					}
				}
				ASSERT(set.has_value());
				for (auto el : elements) {
					rewriter.create<mlir::emitpybytecode::SetAdd>(op.getLoc(), *set, el);
				}
				rewriter.replaceOp(op, *set);
			} else {
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildSet>(
					op, op.getOutput().getType(), op.getElements());
			}

			return success();
		}
	};

	using SetAddOpLowering =
		detail::DirectReplaceLowering<mlir::py::SetAddOp, mlir::emitpybytecode::SetAdd>;

	using BuildStringOpLowering =
		detail::DirectReplaceLowering<mlir::py::BuildStringOp, mlir::emitpybytecode::BuildString>;

	struct FormatValueOpLowering : public mlir::OpRewritePattern<mlir::py::FormatValueOp>
	{
		using OpRewritePattern<mlir::py::FormatValueOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::FormatValueOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::FormatValue>(op,
				op.getOutput().getType(),
				op.getValue(),
				static_cast<uint8_t>(op.getConversion()));

			return success();
		}
	};

	using BuildSliceOpLowering =
		detail::DirectReplaceLowering<mlir::py::BuildSliceOp, mlir::emitpybytecode::BuildSlice>;

}// namespace

void populateCollectionPatterns(mlir::RewritePatternSet &patterns)
{
	auto *ctx = patterns.getContext();
	patterns.add<BuildDictOpLowering,
		DictAddOpLowering,
		BuildListOpLowering,
		ListAppendOpLowering,
		BuildTupleOpLowering,
		BuildSetOpLowering,
		SetAddOpLowering,
		BuildStringOpLowering,
		FormatValueOpLowering,
		BuildSliceOpLowering>(ctx);
}

}// namespace mlir::py
