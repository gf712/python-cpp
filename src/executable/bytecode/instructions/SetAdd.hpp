#pragma once

#include "Instructions.hpp"


class SetAdd final : public Instruction
{
	Register m_set;
	Register m_value;

  public:
	SetAdd(Register set, Register value) : m_set(set), m_value(value) {}

	std::string to_string() const final
	{
		return fmt::format("SET_ADD         r{:<3} r{:<3}", m_set, m_value);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return SET_ADD; }
};