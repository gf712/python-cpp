#pragma once

#include "Instructions.hpp"


class InplaceSub : public Instruction
{
	Register m_lhs;
	Register m_rhs;

  public:
	InplaceSub(Register lhs, Register rhs) : m_lhs(lhs), m_rhs(rhs) {}

	std::string to_string() const final
	{
		return fmt::format("INPLACE_SUB     r{:<3} r{:<3}", m_lhs, m_rhs);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};