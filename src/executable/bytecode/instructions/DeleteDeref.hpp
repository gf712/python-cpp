#pragma once

#include "Instructions.hpp"


class DeleteDeref final : public Instruction
{
	Register m_src;

  public:
	DeleteDeref(Register src) : m_src(src) {}

	std::string to_string() const final { return fmt::format("DELETE_DEREF    f{:<3}", m_src); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return DELETE_DEREF; }
};
