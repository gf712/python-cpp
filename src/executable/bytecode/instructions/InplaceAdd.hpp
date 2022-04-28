#pragma once

#include "Instructions.hpp"


class InplaceAdd : public Instruction
{
	Register m_lhs;
	Register m_rhs;

  public:
	InplaceAdd(Register lhs, Register rhs) : m_lhs(lhs), m_rhs(rhs) {}

	std::string to_string() const final
	{
		return fmt::format("INPLACE_ADD     r{:<3} r{:<3}", m_lhs, m_rhs);
	}

	py::PyResult execute(VirtualMachine &vm, Interpreter &interpreter) const final
	{
		auto &lhs = vm.reg(m_lhs);
		const auto &rhs = vm.reg(m_rhs);
		auto result = add(lhs, rhs, interpreter);
		if (result.is_ok()) { lhs = result.unwrap(); }
		return result;
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final
	{
		return {
			INPLACE_ADD,
			m_lhs,
			m_rhs,
		};
	}
};