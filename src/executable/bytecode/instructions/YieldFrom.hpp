#pragma once

#include "Instructions.hpp"


class YieldFrom final : public Instruction
{
	Register m_dst;
	Register m_receiver;
	Register m_value;

  public:
	YieldFrom(Register dst, Register receiver, Register value)
		: m_dst(dst), m_receiver(receiver), m_value(value)
	{}

	std::string to_string() const final
	{
		return fmt::format("YIELD_FROM      r{:<3} r{:<3} r{:<3}", m_dst, m_receiver, m_value);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;
	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return YIELD_FROM; }
};
