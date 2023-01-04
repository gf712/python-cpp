#pragma once

#include "Instructions.hpp"


class FunctionCallEx final : public Instruction
{
	Register m_function;
	Register m_args;
	Register m_kwargs;
	bool m_expand_args;
	bool m_expand_kwargs;

  public:
	FunctionCallEx(Register function,
		Register args,
		Register kwargs,
		bool expand_args,
		bool expand_kwargs)
		: m_function(function), m_args(args), m_kwargs(kwargs), m_expand_args(expand_args),
		  m_expand_kwargs(expand_kwargs)
	{}

	std::string to_string() const final
	{
		return fmt::format("CALL_EX         r{:<3} r{:<3} r{:<3}", m_function, m_args, m_kwargs);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return FUNCTION_CALL_EX; }
};
