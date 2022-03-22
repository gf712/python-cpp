#pragma once

#include "executable/Program.hpp"

class Function;
class PyFunction;

namespace llvm {
class LLVMContext;
class Module;
}// namespace llvm

class LLVMProgram : public Program
{
	struct InternalConfig;
	struct InteropFunctions;
	std::vector<std::shared_ptr<Function>> m_functions;
	InternalConfig *m_config;
	std::unique_ptr<InteropFunctions> m_interop_functions;

  public:
	LLVMProgram(std::unique_ptr<llvm::Module> &&module,
		std::unique_ptr<llvm::LLVMContext> &&ctx,
		std::string filename,
		std::vector<std::string> argv);

	~LLVMProgram();

	std::string to_string() const override;

	int execute(VirtualMachine *) override;

	py::PyObject *as_pyfunction(const std::string &function_name,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		const std::vector<py::PyCell *> &closure) const override;

  private:
	void create_interop_function(const std::shared_ptr<Function> &,
		const std::string &mangled_name) const;
};