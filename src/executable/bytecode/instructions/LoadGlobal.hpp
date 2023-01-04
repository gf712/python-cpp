#pragma once

#include "Instructions.hpp"


class LoadGlobal final : public Instruction
{
	Register m_destination;
	Register m_object_name;

  public:
	LoadGlobal(Register destination, Register object_name)
		: m_destination(destination), m_object_name(std::move(object_name))
	{}

	std::string to_string() const final
	{
		return fmt::format("LOAD_GLOBAL     r{:<3} \"{}\"", m_destination, m_object_name);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return LOAD_GLOBAL; }
};
