#pragma once

#include "Python/IR/PythonTypes.hpp"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "EmitPythonBytecode/IR/EmitPythonBytecodeOps.h.inc"

#include "EmitPythonBytecode/IR/EmitPythonBytecodeDialect.h.inc"
