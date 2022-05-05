#pragma once

#include "Instructions.hpp"


class WithExceptStart final : public Instruction
{
	Register m_result;
	Register m_exit_method;

  public:
	WithExceptStart(Register result, Register exit_method)
		: m_result(result), m_exit_method(exit_method)
	{}

	std::string to_string() const final
	{
		return fmt::format("WITH_EXCEPT_START r{:<3} r{:<3}", m_result, m_exit_method);
	}

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};