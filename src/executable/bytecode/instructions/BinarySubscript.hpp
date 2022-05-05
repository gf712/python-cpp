#pragma once

#include "Instructions.hpp"

class BinarySubscript final : public Instruction
{
	Register m_dst;
	Register m_src;
	Register m_index;

  public:
	BinarySubscript(Register dst, Register src, Register index)
		: m_dst(dst), m_src(src), m_index(index)
	{}

	std::string to_string() const final
	{
		return fmt::format("BINARY_SUBSCR   r{:<3} r{:<3} r{:<3}", m_dst, m_src, m_index);
	}

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};
