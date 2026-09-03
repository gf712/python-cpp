#include "Conversion/Passes.hpp"
#include "Conversion/PythonToPythonBytecode/PythonToPythonBytecode.hpp"

#include "mlir/Pass/Pass.h"

// After the third-party headers: these name module-owned types.
import py.runtime;

namespace {

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

}// namespace

void mlir::py::registerConversionPasses() { ::registerPasses(); }