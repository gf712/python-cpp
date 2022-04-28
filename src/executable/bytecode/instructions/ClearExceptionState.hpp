#pragma once

#include "Instructions.hpp"

class ClearExceptionState final : public Instruction
{
  public:
	ClearExceptionState() = default;
	std::string to_string() const final { return fmt::format("CLEAR_EXC"); }

	py::PyResult execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};
