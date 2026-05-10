#include "EmitPythonBytecode/IR/EmitPythonBytecode.hpp"

#include "Python/IR/Dialect.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

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
