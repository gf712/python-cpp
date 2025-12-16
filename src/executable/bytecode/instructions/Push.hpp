#pragma once

#include "Instructions.hpp"

class Push final : public Instruction
{
	Register m_source;

  public:
	Push(Register source) : m_source(source) {}
	std::string to_string() const final { return fmt::format("PUSH            r{:<3}", m_source); }
	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return PUSH; }
};
