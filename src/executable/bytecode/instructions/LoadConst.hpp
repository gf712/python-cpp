#pragma once

#include "Instructions.hpp"


class LoadConst final : public Instruction
{
	Register m_destination;
	size_t m_static_value_index;

  public:
	LoadConst(Register destination, size_t static_value_index)
		: m_destination(destination), m_static_value_index(static_value_index)
	{}
	~LoadConst() override {}
	std::string to_string() const final
	{
		return fmt::format("LOAD_CONST      r{:<3} s{:<3}", m_destination, m_static_value_index);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;
	
	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return LOAD_CONST; }
};
