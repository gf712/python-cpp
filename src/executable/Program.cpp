module;
#include <cstdint>

#include "core.hpp"
#include "executable/common.hpp"

module py.runtime;
import py.ast;
import py.codegen;
import std;

// These name py:: types, so they belong in the purview where the implicit
// import of py.runtime has already happened. Including LLVMGenerator.hpp
// unconditionally would attach codegen::LLVMGenerator to py.runtime and pull
// its virtual members into the vtable even though the backend is not built.
#if defined(ENABLE_LLVM_BACKEND) && defined(LLVM_FOUND)
#include "executable/llvm/LLVMGenerator.hpp"
#endif
#include "mlir/compile.hpp"

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
