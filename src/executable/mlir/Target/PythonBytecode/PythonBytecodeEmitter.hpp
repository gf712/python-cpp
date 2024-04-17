#pragma once

#include <memory>

class Program;

namespace mlir {
class Operation;
}

namespace codegen {
std::shared_ptr<Program> translateToPythonBytecode(mlir::Operation *op);
}