#pragma once

#include "Instructions.hpp"


class LoadBuildClass final : public Instruction
{
	Register m_dst;

  public:
	LoadBuildClass(Register dst) : m_dst(dst) {}
	std::string to_string() const final { return fmt::format("LOAD_BUILD_CLASS r{}", m_dst); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;
	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return LOAD_BUILD_CLASS; }
};
