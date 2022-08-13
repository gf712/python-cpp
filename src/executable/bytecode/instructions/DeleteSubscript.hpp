#pragma once

#include "Instructions.hpp"

class DeleteSubscript final : public Instruction
{
	Register m_value;
	Register m_index;

  public:
	DeleteSubscript(Register value, Register index) : m_value(value), m_index(index) {}

	std::string to_string() const final
	{
		return fmt::format("DELETE_SUBSCRIPT r{:<3} r{:<3}", m_value, m_index);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return DELETE_SUBSCRIPT; }
};
