#include "compile.hpp"
#include "Conversion/Passes.hpp"
#include "Conversion/PythonToPythonBytecode/PythonToPythonBytecode.hpp"
#include "Dialect/Python/MLIRGenerator.hpp"
#include "Target/PythonBytecode/PythonBytecodeEmitter.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace compiler::mlir {

std::shared_ptr<Program> compile(std::shared_ptr<ast::Module> node,
	std::vector<std::string> argv,
	compiler::OptimizationLevel)
{
	auto ctx = codegen::Context::create();
	if (!codegen::MLIRGenerator::compile(node, std::move(argv), ctx)) {
		std::cerr << "Failed to compile Python script\n";
		return nullptr;
	}

	::mlir::py::registerConversionPasses();

	::mlir::PassManager pm{ &ctx.ctx() };
	pm.addPass(::mlir::py::createPythonToPythonBytecodePass());
	if (pm.run(ctx.module()).failed()) {
		std::cerr << "Python bytecode MLIR lowering failed\n";
		ctx.module().dump();
		return nullptr;
	}

	return codegen::translateToPythonBytecode(ctx.module());
}

}// namespace compiler::mlir