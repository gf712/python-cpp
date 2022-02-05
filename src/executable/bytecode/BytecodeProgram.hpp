#pragma once

#include "Bytecode.hpp"
#include "executable/Function.hpp"
#include "executable/FunctionBlock.hpp"
#include "executable/Program.hpp"

class BytecodeProgram : public Program
{
	InstructionVector m_instructions;
	std::vector<std::shared_ptr<Function>> m_functions;
	std::shared_ptr<Function> m_main_function;
	std::vector<std::shared_ptr<Program>> m_backends;

  public:
	BytecodeProgram(FunctionBlocks &&func_blocks,
		std::string filename,
		std::vector<std::string> argv);

	auto begin() const
	{
		// FIXME: assumes all functions are bytecode
		ASSERT(m_main_function->backend() == FunctionExecutionBackend::BYTECODE)
		return std::static_pointer_cast<Bytecode>(m_main_function)->begin();
	}

	auto end() const
	{
		// FIXME: assumes all functions are bytecode
		ASSERT(m_main_function->backend() == FunctionExecutionBackend::BYTECODE)
		return std::static_pointer_cast<Bytecode>(m_main_function)->end();
	}

	size_t main_stack_size() const;

	py::PyObject *as_pyfunction(const std::string &function_name,
		const std::vector<std::string> &argnames,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		size_t positional_args_count,
		size_t kwonly_args_count,
		const CodeFlags &flags) const override;

	std::string to_string() const override;

	int execute(VirtualMachine *) override;

	const auto &functions() const { return m_functions; }

	void add_backend(std::shared_ptr<Program>);
};