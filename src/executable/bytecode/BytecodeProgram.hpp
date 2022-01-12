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

	const std::shared_ptr<Function> &function(const std::string &name) const override;

	std::string to_string() const override;

	int execute(VirtualMachine *) override;

	const std::vector<std::shared_ptr<Function>> &functions() const { return m_functions; }
};