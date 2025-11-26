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
}// namespace emitpybytecode
}// namespace mlir

#define GET_OP_CLASSES
#include "EmitPythonBytecode/IR/EmitPythonBytecodeOps.cpp.inc"

// #define GET_TYPEDEF_CLASSES
// #include "EmitPythonBytecode/IR/EmitPythonBytecodeTypes.cpp.inc"
