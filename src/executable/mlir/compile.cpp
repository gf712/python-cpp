#include "compile.hpp"
#include "Conversion/Passes.hpp"
#include "Conversion/PythonToPythonBytecode/PythonToPythonBytecode.hpp"
#include "Dialect/Python/MLIRGenerator.hpp"
#include "Target/PythonBytecode/PythonBytecodeEmitter.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
	// Default PassManager already runs verification after each pass; be
	// explicit so a build that disables it for performance reasons has
	// a single place to flip the switch. Env-gated print-after-all gives
	// a quick way to inspect intermediate IR ('MLIR_PRINT_IR_AFTER_ALL=1
	// ./build/src/python <script>') without recompiling.
	pm.enableVerifier(true);
	if (std::getenv("MLIR_PRINT_IR_AFTER_ALL")) {
		// IR printing requires single-threaded execution so per-pass
		// output isn't interleaved across threads. The pipeline isn't
		// performance-critical for debug runs, so unconditionally
		// switching off multi-threading when the env var is set is fine.
		ctx.ctx().disableMultithreading();
		pm.enableIRPrinting(
			/*shouldPrintBeforePass=*/[](::mlir::Pass *, ::mlir::Operation *) { return false; },
			/*shouldPrintAfterPass=*/[](::mlir::Pass *, ::mlir::Operation *) { return true; },
			/*printModuleScope=*/true,
			/*printAfterOnlyOnChange=*/false,
			/*printAfterOnlyOnFailure=*/false,
			llvm::errs());
	}
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
	// Lower the four region-bearing control-flow ops first, each as its
	// own pass so we can interleave canonicalize + CSE between them. The
	// patterns perform structural surgery (block splits, region inlining,
	// IRMapping clones) and previously ran as part of a single greedy
	// rewrite that couldn't simplify between them.
	//
	// Try and With lower *before* ForLoop and While: a `break`/`continue`
	// inside a try/with body becomes a `br_yield break_/continue_` marker
	// that only the enclosing loop pass can resolve (to the loop's exit /
	// continue target). If the loop lowered first, it would reach into the
	// still-nested try/with region and emit a branch to a block defined in
	// the parent region — invalid IR (a cross-region branch) that sends the
	// later region-DCE into unbounded recursion. Flattening try/with first
	// hoists those markers into the loop body region, so the loop pass
	// resolves them with valid same-region branches.
	pm.addPass(::mlir::py::createConvertTryPass());
	pm.addPass(::mlir::py::createConvertWithPass());
	pm.addPass(::mlir::createCanonicalizerPass());
	pm.addPass(::mlir::createCSEPass());
	pm.addPass(::mlir::py::createConvertForLoopPass());
	pm.addPass(::mlir::py::createConvertWhileLoopPass());
	pm.addPass(::mlir::createCanonicalizerPass());
	pm.addPass(::mlir::createCSEPass());
	pm.addPass(::mlir::py::createPythonToPythonBytecodePass());
	// Post-lowering canonicalize + CSE: dedupes the emitpybytecode.LOAD_CONST
	// ops the lowering and MLIRGenerator emit. Idiomatic Python compiles to
	// many equal None/0/True constants per function.
	pm.addPass(::mlir::createCanonicalizerPass());
	pm.addPass(::mlir::createCSEPass());
	// RemoveDeadValues prunes dead block arguments, dead function results,
	// and operands whose producers are unused. Path (a) from the prior
	// design comment landed: @__hidden_init__ is now marked public, so
	// RemoveDeadValues' processFuncOp skips it ("Function is public or
	// external, skipping") instead of stripping its side-effecting body.
	// User-defined functions remain private — RDV may legitimately drop
	// dead args / results from those.
	//
	// MaterialiseReturnNonePass right after handles the corner case
	// where RDV strips the operand of a func.return inside a non-public
	// function with no callers, leaving a zero-operand return that
	// would violate the bytecode emitter's exactly-one-operand
	// invariant. It re-introduces a None constant + restores the
	// FuncOp's result type.
	pm.addPass(::mlir::createRemoveDeadValuesPass());
	pm.addNestedPass<::mlir::func::FuncOp>(::mlir::py::createMaterialiseReturnNonePass());
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