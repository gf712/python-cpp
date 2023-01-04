#pragma once

#include "Instructions.hpp"
#include <optional>

class MakeFunction : public Instruction
{
	Register m_dst;
	Register m_name;
	size_t m_defaults_size;
	size_t m_defaults_stack_offset;
	size_t m_kw_defaults_size;
	size_t m_kw_defaults_stack_offset;
	std::optional<Register> m_captures_tuple;

  public:
	MakeFunction(Register dst,
		Register function_name,
		size_t defaults_size,
		size_t defaults_stack_offset,
		size_t kw_defaults_size,
		size_t kw_defaults_stack_offset,
		std::optional<Register> captures_tuple)
		: m_dst(dst), m_name(function_name), m_defaults_size(defaults_size),
		  m_defaults_stack_offset(defaults_stack_offset), m_kw_defaults_size(kw_defaults_size),
		  m_kw_defaults_stack_offset(kw_defaults_stack_offset),
		  m_captures_tuple(std::move(captures_tuple))
	{}

	std::string to_string() const final { return fmt::format("MAKE_FUNCTION   ({})", m_name); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return MAKE_FUNCTION; }
};
