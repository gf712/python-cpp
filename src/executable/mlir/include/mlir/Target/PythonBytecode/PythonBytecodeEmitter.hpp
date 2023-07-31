#pragma once

#include <memory>

class BytecodeProgram;

namespace mlir {
class Operation;
}

namespace codegen {
std::shared_ptr<BytecodeProgram> translateToPythonBytecode(mlir::Operation *op);
}