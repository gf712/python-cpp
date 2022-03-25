#pragma once

#include "Bytecode.hpp"
#include "executable/Function.hpp"
#include "executable/FunctionBlock.hpp"
#include "executable/Program.hpp"
#include "runtime/forward.hpp"

class BytecodeProgram : public Program
{
	InstructionVector m_instructions;
	std::vector<py::PyCode *> m_functions;
	py::PyCode *m_main_function;
	std::vector<std::shared_ptr<Program>> m_backends;

  public:
	BytecodeProgram(FunctionBlocks &&func_blocks,
		std::string filename,
		std::vector<std::string> argv);

	std::vector<View>::const_iterator begin() const;

	std::vector<View>::const_iterator end() const;

	size_t main_stack_size() const;

	py::PyObject *as_pyfunction(const std::string &function_name,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		const std::vector<py::PyCell *> &closure) const override;

	std::string to_string() const override;

	int execute(VirtualMachine *) override;

	const auto &functions() const { return m_functions; }

	const auto &main_function() const { return m_main_function; }

	void add_backend(std::shared_ptr<Program>);

	void visit_functions(Cell::Visitor &) const override;
};