#include "EmitPythonBytecode/IR/EmitPythonBytecode.hpp"

#include "Python/Dialect.hpp"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

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

		// 		addTypes<
		// #define GET_TYPEDEF_LIST
		// #include "EmitPythonBytecode/IR/EmitPythonBytecodeTypes.cpp.inc"
		// 			>();
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
}// namespace emitpybytecode
}// namespace mlir

#define GET_OP_CLASSES
#include "EmitPythonBytecode/IR/EmitPythonBytecodeOps.cpp.inc"

// #define GET_TYPEDEF_CLASSES
// #include "EmitPythonBytecode/IR/EmitPythonBytecodeTypes.cpp.inc"
