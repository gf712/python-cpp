#pragma once

#include "Instructions.hpp"


class BuildDict final : public Instruction
{
	Register m_dst;
	size_t m_size;
	size_t m_stack_offset;

  public:
	BuildDict(Register dst, size_t size, size_t stack_offset)
		: m_dst(dst), m_size(size), m_stack_offset(stack_offset)
	{}

	std::string to_string() const final
	{
		return fmt::format(
			"BUILD_DICT      r{:<3} (size={}, offset={})", m_dst, m_size, m_stack_offset);
	}

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};