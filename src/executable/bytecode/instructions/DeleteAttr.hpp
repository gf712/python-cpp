#pragma once

#include "Instructions.hpp"


class DeleteAttr final : public Instruction
{
	Register m_self;
	Register m_attr_name;

  public:
	DeleteAttr(Register self, Register attr_name) : m_self(self), m_attr_name(attr_name) {}

	std::string to_string() const final
	{
		return fmt::format("DELETE_ATTR     r{:<3} ({})", m_self, m_attr_name);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return DELETE_ATTR; }
};
