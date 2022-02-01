#pragma once

#include "Instructions.hpp"


class ListExtend final : public Instruction
{
	Register m_list;
	Register m_value;

  public:
	ListExtend(Register list, Register value) : m_list(list), m_value(value) {}

	std::string to_string() const final
	{
		return fmt::format("LIST_EXT        r{:<3} r{:<3}", m_list, m_value);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};