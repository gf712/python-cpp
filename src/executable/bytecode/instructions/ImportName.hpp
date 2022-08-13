#pragma once

#include "Instructions.hpp"

#include <numeric>

class ImportName : public Instruction
{
	Register m_destination;
	Register m_name;
	Register m_from_list;
	Register m_level;

  public:
	ImportName(Register dst, Register name, Register from_list, Register level)
		: m_destination(dst), m_name(name), m_from_list(from_list), m_level(level)
	{}

	std::string to_string() const final
	{
		return fmt::format("IMPORT_NAME     r{:<3} {:<3}", m_destination, m_name);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return IMPORT_NAME; }
};