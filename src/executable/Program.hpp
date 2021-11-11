#pragma once

#include "FunctionBlock.hpp"
#include "bytecode/Bytecode.hpp"
#include "utilities.hpp"

class Program : NonCopyable
{
	std::string m_filename;
	std::vector<std::string> m_argv;
	InstructionVector m_instructions;
	std::vector<std::shared_ptr<Function>> m_functions;
	std::shared_ptr<Function> m_main_function;

  public:
	Program(FunctionBlocks &&func_blocks, std::string filename, std::vector<std::string> argv);

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
	size_t main_stack_size() const { return m_main_function->registers_needed(); }
	const std::string &filename() const { return m_filename; }
	const std::vector<std::string> &argv() const { return m_argv; }

	void set_filename(std::string filename) { m_filename = std::move(filename); }

	const std::shared_ptr<Function> &function(size_t idx)
	{
		if (idx == 0) {
			ASSERT(m_main_function)
			return m_main_function;
		}
		ASSERT(idx <= m_functions.size())
		return m_functions[idx - 1];
	}

	const std::vector<std::shared_ptr<Function>> &functions() const { return m_functions; }

	std::string to_string() const;
};
