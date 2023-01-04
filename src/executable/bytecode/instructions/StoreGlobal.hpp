#pragma once

#include "Instructions.hpp"


class StoreGlobal final : public Instruction
{
	Register m_object_name;
	Register m_source;

  public:
	StoreGlobal(Register object_name, Register source)
		: m_object_name(object_name), m_source(source)
	{}
	std::string to_string() const final
	{
		return fmt::format("STORE_GLOBAL    {:<3} r{:<3}", m_object_name, m_source);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return STORE_GLOBAL; }
};
