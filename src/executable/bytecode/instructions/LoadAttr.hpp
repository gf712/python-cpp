#pragma once

#include "Instructions.hpp"


class LoadAttr final : public Instruction
{
	Register m_destination;
	Register m_value_source;
	const std::string m_attr_name;

  public:
	LoadAttr(Register destination, Register value_source, std::string attr_name)
		: m_destination(destination), m_value_source(value_source),
		  m_attr_name(std::move(attr_name))
	{}

	std::string to_string() const final
	{
		return fmt::format(
			"LOAD_ATTR       r{:<3} r{:<3} ({})", m_destination, m_value_source, m_attr_name);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return LOAD_ATTR; }
};