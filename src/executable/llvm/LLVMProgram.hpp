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

	py::PyObject *as_pyfunction(const std::string &function_name,
		const std::vector<std::string> &argnames,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		size_t positional_args_count,
		size_t kwonly_args_count,
		const py::PyCode::CodeFlags &flags) const override;
};