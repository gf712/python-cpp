#include "Conversion/PythonToPythonBytecode/LoweringHelpers.hpp"
#include "Conversion/PythonToPythonBytecode/PatternPopulators.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"

#include "utilities.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"

#include <algorithm>
#include <vector>

namespace mlir::py {
namespace {

	struct CallFunctionLowering : public mlir::OpRewritePattern<py::FunctionCallOp>
	{
		using OpRewritePattern<py::FunctionCallOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(py::FunctionCallOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto callee = op.getCallee();
			auto args = op.getArgs();

			if (op.getRequiresArgsExpansion() || op.getRequiresKwargsExpansion()) {
				ASSERT(args.size() <= 1);
				ASSERT(op.getKwargs().size() <= 1);
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::FunctionCallExOp>(op,
					op.getOutput().getType(),
					callee,
					op.getRequiresArgsExpansion() ? args.front() : nullptr,
					op.getRequiresKwargsExpansion() ? op.getKwargs().front() : nullptr);
			} else if (!op.getKeywords().empty()) {
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::FunctionCallWithKeywordsOp>(
					op, op.getOutput().getType(), callee, args, op.getKeywords(), op.getKwargs());
			} else {
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::FunctionCallOp>(
					op, op.getOutput().getType(), callee, args);
			}

			return success();
		}
	};

	// py.return -> func.return. py.return relaxes func.return's
	// HasParent<FuncOp> constraint so that return statements emitted
	// inside py.try / py.with regions don't trip the verifier
	// pre-lowering. By the time this pattern fires, the surrounding
	// region ops have been flattened into the enclosing func.func's
	// body, so func.return is well-formed.
	struct ReturnOpLowering : public mlir::OpRewritePattern<mlir::py::ReturnOp>
	{
		using OpRewritePattern<mlir::py::ReturnOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::ReturnOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, op.getValue());
			return mlir::success();
		}
	};

	struct MakeFunctionOpLowering : public mlir::OpRewritePattern<mlir::py::MakeFunctionOp>
	{
		using OpRewritePattern<mlir::py::MakeFunctionOp>::OpRewritePattern;
		mlir::LogicalResult matchAndRewrite(mlir::py::MakeFunctionOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto module = op->getParentOfType<mlir::ModuleOp>();
			auto function_definition = module.lookupSymbol(op.getFunctionName());
			ASSERT(function_definition);
			ASSERT(mlir::isa<mlir::func::FuncOp>(*function_definition));

			auto sym_name = rewriter.create<mlir::emitpybytecode::ConstantOp>(op.getLoc(),
				mlir::py::PyObjectType::get(rewriter.getContext()),
				rewriter.getStringAttr(op.getFunctionName()));

			auto captures_tuple = [&]() -> mlir::Value {
				if (op.getCaptures().empty()) { return nullptr; }
				std::vector<mlir::Value> captures_vec;
				for (auto attr : op.getCaptures()) {
					auto name = mlir::cast<mlir::StringAttr>(attr).getValue();
					captures_vec.push_back(rewriter.create<mlir::emitpybytecode::LoadClosureOp>(
						op.getLoc(), mlir::py::PyObjectType::get(getContext()), name));
				}
				return rewriter.create<mlir::emitpybytecode::BuildTuple>(
					op.getLoc(), mlir::py::PyObjectType::get(getContext()), captures_vec);
			}();
			rewriter.replaceOpWithNewOp<mlir::emitpybytecode::MakeFunction>(op,
				mlir::py::PyObjectType::get(rewriter.getContext()),
				sym_name,
				op.getDefaults(),
				op.getKwDefaults(),
				captures_tuple);

			return success();
		}
	};

	struct FuncOpLowering : public mlir::OpRewritePattern<mlir::func::FuncOp>
	{
		using OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			if (op.isPrivate()) { return success(); }
			populate_arguments(op, rewriter);
			return success();
		}

		void populate_arguments(mlir::func::FuncOp &op, mlir::OpBuilder &builder) const
		{
			for (size_t i = 0; i < op.getNumArguments(); ++i) {
				auto arg_name = op.getArgAttr(i, "llvm.name");
				ASSERT(arg_name);
				detail::add_identifier_to(
					op, "locals", mlir::cast<mlir::StringAttr>(arg_name).getValue(), builder);
			}
		}
	};

	struct ClassDefinitionOpLowering : public mlir::OpRewritePattern<mlir::py::ClassDefinitionOp>
	{
		using OpRewritePattern<mlir::py::ClassDefinitionOp>::OpRewritePattern;

		mlir::LogicalResult matchAndRewrite(mlir::py::ClassDefinitionOp op,
			mlir::PatternRewriter &rewriter) const final
		{
			auto module = op->getParentOfType<mlir::ModuleOp>();
			rewriter.setInsertionPointToEnd(module.getBody());

			auto func_type = rewriter.getFunctionType(mlir::TypeRange{},
				mlir::TypeRange{ mlir::py::PyObjectType::get(rewriter.getContext()) });
			auto class_fn_definition = rewriter.create<mlir::func::FuncOp>(op.getLoc(),
				op.getMangledName(),
				func_type,
				mlir::ArrayRef<mlir::NamedAttribute>{},
				mlir::ArrayRef<mlir::DictionaryAttr>{});

			class_fn_definition->setAttr("is_class", rewriter.getBoolAttr(true));

			if (auto cellvars = op->getAttrOfType<mlir::ArrayAttr>("cellvars")) {
				auto cell_names = cellvars.getValue();
				if (std::find_if(cell_names.begin(),
						cell_names.end(),
						[](mlir::Attribute name) {
							return mlir::cast<mlir::StringAttr>(name) == "__class__";
						})
					!= cell_names.end()) {

					mlir::py::ClassReturnOp return_op;
					op.getBody().walk<WalkOrder::PreOrder>([&return_op](mlir::Operation *child_op) {
						if (mlir::isa<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(child_op)) {
							return WalkResult::skip();
						}
						if (auto cr = mlir::dyn_cast<mlir::py::ClassReturnOp>(child_op)) {
							return_op = cr;
							return WalkResult::interrupt();
						}
						return WalkResult::advance();
					});
					ASSERT(return_op);
					ASSERT(return_op->getParentOp() == op.getOperation());
					ASSERT(return_op.getValue().getDefiningOp());
					rewriter.setInsertionPoint(return_op.getValue().getDefiningOp());
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadClosureOp>(
						return_op.getValue().getDefiningOp(),
						mlir::py::PyObjectType::get(getContext()),
						mlir::StringRef{ "__class__" });
				}
			}

			auto attr = class_fn_definition->getAttrs().vec();
			attr.insert(attr.end(), op->getAttrs().begin(), op->getAttrs().end());
			class_fn_definition->setAttrs(attr);

			// Convert all py.class_return ops in the class body to
			// func.return so that the body, once inlined into the
			// synthesised func.func, has a valid terminator.
			op.getBody().walk([&rewriter](mlir::py::ClassReturnOp cr) {
				rewriter.setInsertionPoint(cr);
				rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(
					cr, mlir::ValueRange{ cr.getValue() });
			});

			auto *end = class_fn_definition.addEntryBlock();
			rewriter.setInsertionPointToStart(end);
			rewriter.inlineRegionBefore(op.getBody(), &class_fn_definition.getBody().front());
			rewriter.eraseBlock(end);

			rewriter.setInsertionPoint(op);
			auto class_name = rewriter.create<mlir::emitpybytecode::ConstantOp>(op.getLoc(),
				mlir::py::PyObjectType::get(rewriter.getContext()),
				rewriter.getStringAttr(op.getMangledName()));

			auto captures_tuple = [&]() -> mlir::Value {
				if (op.getCaptures().empty()) { return {}; }
				std::vector<mlir::Value> captures_vec;
				for (auto attr : op.getCaptures()) {
					auto name = mlir::cast<mlir::StringAttr>(attr).getValue();
					captures_vec.push_back(rewriter.create<mlir::emitpybytecode::LoadClosureOp>(
						op.getLoc(), mlir::py::PyObjectType::get(getContext()), name));
				}
				return rewriter.create<mlir::emitpybytecode::BuildTuple>(
					op.getLoc(), mlir::py::PyObjectType::get(getContext()), captures_vec);
			}();

			auto class_fn = rewriter.create<mlir::emitpybytecode::MakeFunction>(op.getLoc(),
				mlir::py::PyObjectType::get(rewriter.getContext()),
				class_name,
				mlir::ValueRange{},
				mlir::ValueRange{},
				captures_tuple);

			auto class_builder = rewriter.create<mlir::emitpybytecode::LoadBuildClass>(
				op.getLoc(), mlir::py::PyObjectType::get(rewriter.getContext()));
			std::vector<mlir::Value> args{ class_fn, class_name };
			args.insert(args.end(), op.getBases().begin(), op.getBases().end());
			rewriter.replaceOpWithNewOp<py::FunctionCallOp>(op,
				op.getOutput().getType(),
				class_builder,
				args,
				op.getKeywords(),
				op.getKwargs(),
				false,
				false);

			return success();
		}
	};

}// namespace

void populateFunctionPatterns(mlir::RewritePatternSet &patterns)
{
	auto *ctx = patterns.getContext();
	patterns.add<CallFunctionLowering,
		ReturnOpLowering,
		MakeFunctionOpLowering,
		FuncOpLowering,
		ClassDefinitionOpLowering>(ctx);
}

}// namespace mlir::py
