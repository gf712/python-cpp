#pragma once

#include "Instructions.hpp"

class Pop final : public Instruction
{
	bool m_discard{ true };
	Register m_dst;

  public:
	Pop() {}
	Pop(Register dst) : m_discard(false), m_dst(dst) {}
	std::string to_string() const final { return "POP"; }
	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return POP; }
};
