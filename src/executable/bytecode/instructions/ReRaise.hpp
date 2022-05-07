#pragma once

#include "Instructions.hpp"

class ReRaise final : public Instruction
{
  public:
	ReRaise() = default;
	std::string to_string() const final { return std::string("RERAISE"); }

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return RERAISE; }
};
