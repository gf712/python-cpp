#pragma once

#include <memory>

namespace mlir {

class Pass;

namespace py {
	std::unique_ptr<Pass> createPythonToPythonBytecodePass();
}

}// namespace mlir
