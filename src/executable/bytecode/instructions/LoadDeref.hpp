#pragma once

#include "Instructions.hpp"


class LoadDeref final : public Instruction
{
	Register m_destination;
	Register m_source;

  public:
	LoadDeref(Register destination, Register source) : m_destination(destination), m_source(source)
	{}

	std::string to_string() const final
	{
		return fmt::format("LOAD_DEREF      r{:<3} f{:<3}", m_destination, m_source);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return LOAD_DEREF; }
};