#pragma once

#include "Instructions.hpp"


class ListToTuple final : public Instruction
{
	Register m_tuple;
	Register m_list;

  public:
	ListToTuple(Register tuple, Register list) : m_tuple(tuple), m_list(list) {}

	std::string to_string() const final
	{
		return fmt::format("LIST_TO_TUPLE   r{:<3} r{:<3}", m_tuple, m_list);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};