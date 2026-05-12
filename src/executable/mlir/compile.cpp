#include "compile.hpp"
#include "Conversion/Passes.hpp"
#include "Conversion/PythonToPythonBytecode/PythonToPythonBytecode.hpp"
#include "Dialect/Python/MLIRGenerator.hpp"
#include "Target/PythonBytecode/PythonBytecodeEmitter.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "llvm/Support/GraphWriter.h"


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
	// Pre-lowering canonicalize + CSE on the Python dialect. Now safe to run
	// because Load* declares a MemWrite on PythonExceptionStateResource
	// (preventing DCE of unused loads whose may-raise side effect is
	// observable), ForLoopOp's orelse is AnyRegion (so canonicalize can drop
	// the empty entry block), and ClassDefinitionOp uses py.class_return
	// (with HasParent<ClassDefinitionOp>) as its body terminator instead of
	// func.return (which would trip the func dialect's parent-type
	// verifier).
	pm.addPass(::mlir::createCanonicalizerPass());
	pm.addPass(::mlir::createCSEPass());
	pm.addPass(::mlir::py::createPythonToPythonBytecodePass());
	// Post-lowering canonicalize + CSE: dedupes the emitpybytecode.LOAD_CONST
	// ops the lowering and MLIRGenerator emit. Idiomatic Python compiles to
	// many equal None/0/True constants per function.
	pm.addPass(::mlir::createCanonicalizerPass());
	pm.addPass(::mlir::createCSEPass());
	// pm.addPass(::mlir::createRemoveDeadValuesPass());
	// {
	// 	int fd;
	// 	std::string filename = llvm::createGraphFilename("python", fd);
	// 	{
	// 		llvm::raw_fd_ostream os(fd, /*shouldClose=*/true);
	// 		if (fd == -1) {
	// 			llvm::errs() << "error opening file '" << filename << "' for writing\n";
	// 			return nullptr;
	// 		}
	// 		pm.addPass(::mlir::createPrintOpGraphPass(os));
	// 	}
	// 	llvm::DisplayGraph(filename, /*wait=*/false, llvm::GraphProgram::DOT);
	// }
	if (pm.run(ctx.module()).failed()) {
		std::cerr << "Python bytecode MLIR lowering failed\n";
		ctx.module().dump();
		return nullptr;
	}

	return codegen::translateToPythonBytecode(ctx.module());
}

}// namespace compiler::mlir