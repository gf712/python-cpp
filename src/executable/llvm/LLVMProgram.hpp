#pragma once

#include "executable/Program.hpp"

namespace llvm {
class LLVMContext;
class Module;
}// namespace llvm

class Function;

class LLVMProgram : public Program
{
	// order matters here, since the destructor of LLVMContext relies on Module
	std::unique_ptr<llvm::LLVMContext> m_ctx;
	std::unique_ptr<llvm::Module> m_module;
	std::vector<std::shared_ptr<Function>> m_functions;

  public:
	LLVMProgram(std::unique_ptr<llvm::Module> &&module,
		std::unique_ptr<llvm::LLVMContext> &&ctx,
		std::string filename,
		std::vector<std::string> argv);

	std::string to_string() const override;

	int execute(VirtualMachine *) override;

	const std::shared_ptr<Function> &function(const std::string &) const override;
};