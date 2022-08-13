#pragma once

#include "Instructions.hpp"

class ClearTopCleanup final : public Instruction
{
  public:
	ClearTopCleanup() = default;
	std::string to_string() const final { return fmt::format("CLEAR_TOP_CLEANUP"); }

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return CLEAR_EXCEPTION_STATE; }
};
