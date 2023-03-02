#pragma once

#include "Instructions.hpp"

class ImportStar : public Instruction
{
	Register m_src;

  public:
	ImportStar(Register src) : m_src(src) {}

	std::string to_string() const final { return fmt::format("IMPORT_STAR     r{:<3}", m_src); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return IMPORT_STAR; }
};
