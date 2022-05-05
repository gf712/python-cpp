#pragma once

#include "Instructions.hpp"

class DeleteName final : public Instruction
{
	Register m_name;

  public:
	DeleteName(Register name) : m_name(name) {}

	std::string to_string() const final { return fmt::format("DELETE_NAME     r{:<3}", m_name); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};
