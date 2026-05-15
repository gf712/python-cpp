#include "Conversion/PythonToPythonBytecode/LoweringHelpers.hpp"
#include "Conversion/PythonToPythonBytecode/PatternPopulators.hpp"

#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"

#include "mlir/IR/PatternMatch.h"

namespace mlir::py {
namespace {

	using LoadAttributeOpLowering = detail::DirectReplaceLowering<mlir::py::LoadAttributeOp,
		mlir::emitpybytecode::LoadAttribute>;

	using DeleteAttributeOpLowering = detail::DirectReplaceLowering<mlir::py::DeleteAttributeOp,
		mlir::emitpybytecode::DeleteAttribute>;

	using LoadMethodOpLowering = detail::DirectReplaceRegisterName<mlir::py::LoadMethodOp,
		mlir::emitpybytecode::LoadMethod,
		&mlir::py::LoadMethodOp::getMethodName>;

	using BinarySubscriptOpLowering = detail::DirectReplaceLowering<mlir::py::BinarySubscriptOp,
		mlir::emitpybytecode::BinarySubscript>;

	using StoreSubscriptOpLowering = detail::DirectReplaceLowering<mlir::py::StoreSubscriptOp,
		mlir::emitpybytecode::StoreSubscript>;

	using DeleteSubscriptOpLowering = detail::DirectReplaceLowering<mlir::py::DeleteSubscriptOp,
		mlir::emitpybytecode::DeleteSubscript>;

	using StoreAttributeOpLowering = detail::DirectReplaceRegisterName<mlir::py::StoreAttributeOp,
		mlir::emitpybytecode::StoreAttribute,
		&mlir::py::StoreAttributeOp::getAttr>;

}// namespace

void populateAttributeSubscriptPatterns(mlir::RewritePatternSet &patterns)
{
	auto *ctx = patterns.getContext();
	patterns.add<LoadAttributeOpLowering,
		DeleteAttributeOpLowering,
		LoadMethodOpLowering,
		BinarySubscriptOpLowering,
		StoreSubscriptOpLowering,
		DeleteSubscriptOpLowering,
		StoreAttributeOpLowering>(ctx);
}

}// namespace mlir::py
