#pragma once

#include "Instructions.hpp"


class UnpackSequence final : public Instruction
{
	std::vector<Register> m_destination;
	Register m_source;

  public:
	UnpackSequence(std::vector<Register> destination, Register source)
		: m_destination(std::move(destination)), m_source(source)
	{}
	std::string to_string() const final
	{
		return fmt::format("UNPACK_SEQUENCE {} r{:<3}", m_destination.size(), m_source);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return UNPACK_SEQUENCE; }
};
