// python-mlir-opt: a minimal mlir-opt-like tool that knows about the
// Python and EmitPythonBytecode dialects (plus the standard MLIR
// dialects the conversion patterns produce) and all of this
// project's passes. Used by the lit test suite to round-trip ops,
// run individual passes, and FileCheck their output - none of which
// is possible with the stock /usr/bin/mlir-opt because it has no
// awareness of our dialects.

#include "Conversion/Passes.hpp"
#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/Dialect.hpp"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv)
{
	mlir::DialectRegistry registry;
	mlir::registerAllDialects(registry);
	mlir::registerAllExtensions(registry);
	registry.insert<mlir::py::PythonDialect, mlir::emitpybytecode::EmitPythonBytecodeDialect>();

	mlir::registerAllPasses();
	mlir::py::registerConversionPasses();

	return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "python-mlir-opt\n", registry));
}
