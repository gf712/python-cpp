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
				// Register-pressure workaround for dict literals larger
				// than ~10 entries. A single BuildDict consuming N keys
				// + N values forces N+N live values to coexist in
				// registers right before the call, which the linear-
				// scan allocator (see LinearScanRegisterAllocation in
				// Target/PythonBytecode/) handles by spilling - and
				// spills are expensive because emitpybytecode has no
				// stack slots, only register-to-register moves and
				// Push/Pop. Emit an empty BuildDict and stream each kv
				// pair via DictAdd directly after its value is computed
				// instead, so only 3 live values (dict, key, value) are
				// needed at any point. The threshold is empirical; the
				// principled fix is to do register allocation as an
				// MLIR pass before bytecode emission (plan step 19) so
				// the lowering can pick a strategy based on actual
				// pressure feedback.
				if (op.getValues().size() > 10) {
					auto keys = op.getKeys();
					auto values = op.getValues();
					rewriter.setInsertionPointAfterValue(keys.front());
					auto result = rewriter.create<mlir::emitpybytecode::BuildDict>(op->getLoc(),
						op.getOutput().getType(),
						mlir::ValueRange{},
						mlir::ValueRange{});

					for (auto [key, value] : llvm::zip(keys, values)) {
						rewriter.setInsertionPointAfterValue(value);
						rewriter.create<mlir::emitpybytecode::DictAdd>(
							op.getLoc(), result, key, value);
					}
					rewriter.replaceOp(op, result);
				} else {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildDict>(
						op, op.getOutput().getType(), op.getKeys(), op.getValues());
				}
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
			auto known_at_compiletime = [op](mlir::Value element) -> bool {
				return element.getDefiningOp()
					   && (mlir::isa<mlir::py::ConstantOp>(element.getDefiningOp())
						   || mlir::isa<mlir::emitpybytecode::ConstantOp>(element.getDefiningOp()));
			};
			if (std::any_of(requires_expansion.begin(),
					requires_expansion.end(),
					[](const auto &el) { return el == 1; })) {
				auto list = build_list_with_expansion(rewriter,
					op.getLoc(),
					op.getOutput().getType(),
					op.getElements(),
					requires_expansion);
				rewriter.replaceOp(op, list);
			} else if (std::all_of(op.getElements().begin(),
						   op.getElements().end(),
						   known_at_compiletime)) {
				// Same register-pressure motivation as the BuildDict
				// >10 branch above: an all-constants list literal
				// would otherwise tie up N registers waiting for the
				// BuildList consumer. Bake the elements into a tuple
				// Attribute and emit a single ListExtend from the
				// constant - 2 registers live (list, tuple), not N.
				// The principled fix is to expose register-pressure
				// data to the lowering via a real MLIR allocation
				// pass (plan step 19).
				std::vector<mlir::Attribute> elements;
				elements.reserve(op.getElements().size());
				for (const auto &el : op.getElements()) {
					if (el.getDefiningOp<mlir::py::ConstantOp>()) {
						elements.push_back(el.getDefiningOp<mlir::py::ConstantOp>().getValue());
					} else {
						ASSERT(el.getDefiningOp<mlir::emitpybytecode::ConstantOp>());
						elements.push_back(
							el.getDefiningOp<mlir::emitpybytecode::ConstantOp>().getValue());
					}
				}
				auto loc = op.getLoc();
				auto output_type = op.getOutput().getType();
				auto list = rewriter.create<mlir::emitpybytecode::BuildList>(
					loc, output_type, ::mlir::ValueRange{});
				auto tuple = rewriter.create<mlir::emitpybytecode::ConstantOp>(
					loc, output_type, mlir::ArrayAttr::get(getContext(), elements));
				rewriter.create<mlir::emitpybytecode::ListExtend>(loc, list, tuple);
				rewriter.replaceOp(op, list);
			} else {
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
