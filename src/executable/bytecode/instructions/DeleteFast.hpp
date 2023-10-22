#pragma once

#include "Instructions.hpp"

class DeleteFast final : public Instruction
{
	Register m_stack_index;

  public:
	DeleteFast(Register stack_index) : m_stack_index(stack_index) {}

	std::string to_string() const final { return fmt::format("DELETE_FAST     {}", m_stack_index); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return DELETE_FAST; }
};
