#include "Conversion/Passes.hpp"
#include "Conversion/PythonToPythonBytecode/PythonToPythonBytecode.hpp"

#include "mlir/Pass/Pass.h"

namespace {

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

}// namespace

void mlir::py::registerConversionPasses() { ::registerPasses(); }