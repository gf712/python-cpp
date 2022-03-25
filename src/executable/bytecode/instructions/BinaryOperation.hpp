#pragma once

#include "Instructions.hpp"

class BinaryOperation : public Instruction
{
  public:
	enum class Operation {
		PLUS = 0,
		MINUS = 1,
		MODULO = 2,
		MULTIPLY = 3,
		EXP = 4,
		SLASH = 5,
		FLOORDIV = 6,
		LEFTSHIFT = 7,
		RIGHTSHIFT = 8,
	};

  private:
	Register m_destination;
	Register m_lhs;
	Register m_rhs;
	Operation m_operation;

  public:
	BinaryOperation(Register dst, Register lhs, Register rhs, Operation op)
		: m_destination(dst), m_lhs(lhs), m_rhs(rhs), m_operation(op)
	{}

	std::string to_string() const final
	{
		return fmt::format(
			"BINARY_OP       r{:<3} r{:<3} r{:<3} ({})", m_destination, m_lhs, m_rhs, m_operation);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
