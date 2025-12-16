#include "Program.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"
#include "executable/llvm/LLVMGenerator.hpp"
#include "executable/mlir/Dialect/Python/MLIRGenerator.hpp"
#include "mlir/compile.hpp"
#include "utilities.hpp"


Program::Program(std::string &&filename, std::vector<std::string> &&argv)
	: m_filename(std::move(filename)), m_argv(std::move(argv))
{}

namespace compiler {
std::shared_ptr<Program> compile(std::shared_ptr<ast::Module> node,
	std::vector<std::string> argv,
	Backend backend,
	OptimizationLevel lvl)
{
	switch (backend) {

	case Backend::BYTECODE_GENERATOR:
		return codegen::BytecodeGenerator::compile(node, std::move(argv), lvl);
	case Backend::LLVM: {
#if defined(ENABLE_LLVM_BACKEND) && defined(LLVM_FOUND)
		return codegen::LLVMGenerator::compile(node, std::move(argv), lvl);
#else
		std::cerr << "LLVM backend unavailable\n";
		return nullptr;
#endif
	}
	case Backend::MLIR: {
		return compiler::mlir::compile(node, std::move(argv), lvl);
	}
	}
	ASSERT_NOT_REACHED();
}
}// namespace compiler