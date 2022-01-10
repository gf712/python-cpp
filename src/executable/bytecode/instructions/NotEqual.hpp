#pragma once

#include "Instructions.hpp"

class NotEqual final : public Instruction
{
	Register m_dst;
	Register m_lhs;
	Register m_rhs;

  public:
	NotEqual(Register dst, Register lhs, Register rhs) : m_dst(dst), m_lhs(lhs), m_rhs(rhs) {}

	std::string to_string() const final
	{
		return fmt::format("NOT_EQUAL        r{:<3} r{:<3} r{:<3}", m_dst, m_lhs, m_rhs);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};