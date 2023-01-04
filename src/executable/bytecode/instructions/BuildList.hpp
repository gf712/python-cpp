#pragma once

#include "Instructions.hpp"


class BuildList final : public Instruction
{
	Register m_dst;
	size_t m_size;
	size_t m_stack_offset;

  public:
	BuildList(Register dst, size_t size, size_t stack_offset)
		: m_dst(dst), m_size(size), m_stack_offset(stack_offset)
	{}

	std::string to_string() const final
	{
		return fmt::format(
			"BUILD_LIST      r{:<3} (size={}, offset={})", m_dst, m_size, m_stack_offset);
	}

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return BUILD_LIST; }
};
