#pragma once

#include "Instructions.hpp"

class ClearExceptionState final : public Instruction
{
  public:
	ClearExceptionState() = default;
	std::string to_string() const final { return fmt::format("CLEAR_EXC"); }

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
