#pragma once

#include "Instructions.hpp"


class YieldValue final : public Instruction
{
	Register m_source;

  public:
	YieldValue(Register source) : m_source(source) {}
	~YieldValue() override {}
	std::string to_string() const final { return fmt::format("YIELD_VALUE     r{:<3}", m_source); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;
	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return YIELD_VALUE; }
};