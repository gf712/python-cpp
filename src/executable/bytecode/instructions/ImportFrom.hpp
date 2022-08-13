#pragma once

#include "Instructions.hpp"

class ImportFrom : public Instruction
{
	Register m_destination;
	Register m_name;
	Register m_from;

  public:
	ImportFrom(Register dst, Register name, Register from)
		: m_destination(dst), m_name(name), m_from(from)
	{}

	std::string to_string() const final
	{
		return fmt::format("IMPORT_FROM     r{:<3} {:<3}", m_destination, m_name);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return IMPORT_FROM; }
};