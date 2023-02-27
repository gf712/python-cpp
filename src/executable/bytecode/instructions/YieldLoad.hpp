#pragma once

#include "Instructions.hpp"


class YieldLoad final : public Instruction
{
	Register m_dst;

  public:
	YieldLoad(Register dst) : m_dst(dst) {}

	std::string to_string() const final { return fmt::format("YIELD_LOAD      r{:<3}", m_dst); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;
	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return YIELD_LOAD; }
};
