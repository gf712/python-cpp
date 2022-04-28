#pragma once

#include "Instructions.hpp"


class Unary final : public Instruction
{
  public:
	enum class Operation {
		POSITIVE = 0,
		NEGATIVE = 1,
		INVERT = 2,
		NOT = 3,
	};

	Register m_destination;
	Register m_source;
	Operation m_operation;

  public:
	Unary(Register destination, Register source, Operation operation)
		: m_destination(destination), m_source(source), m_operation(operation)
	{}

	std::string to_string() const final
	{
		return fmt::format("UNARY  r{:<3} r{:<3}", m_destination, m_source, m_operation);
	}

	py::PyResult execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};
