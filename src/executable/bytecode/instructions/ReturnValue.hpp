#pragma once

#include "Instructions.hpp"


class ReturnValue final : public Instruction
{
	Register m_source;

  public:
	ReturnValue(Register source) : m_source(source) {}
	~ReturnValue() override {}
	std::string to_string() const final { return fmt::format("RETURN_VALUE    r{:<3}", m_source); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;
	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	Register source() const { return m_source; }

	uint8_t id() const final { return RETURN_VALUE; }
};
