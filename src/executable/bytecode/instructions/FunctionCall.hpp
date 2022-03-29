#pragma once

#include "Instructions.hpp"

class FunctionCall final : public Instruction
{
	Register m_function_name;
	size_t m_size;
	size_t m_stack_offset;

  public:
	FunctionCall(Register function_name, size_t size, size_t stack_offset)
		: m_function_name(function_name), m_size(size), m_stack_offset(stack_offset)
	{}

	std::string to_string() const final
	{
		return fmt::format("CALL            r{:<3} ({})", m_function_name, m_size);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};
