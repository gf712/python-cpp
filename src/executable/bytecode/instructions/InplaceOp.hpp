#pragma once

#include "Instructions.hpp"


class InplaceOp : public Instruction
{
  public:
	enum class Operation : uint8_t {
		PLUS = 0,
		MINUS = 1,
		MODULO = 2,
		MULTIPLY = 3,
		EXP = 4,
		SLASH = 5,
		FLOORDIV = 6,
		LEFTSHIFT = 7,
		RIGHTSHIFT = 8,
		AND = 9,
		OR = 10,
		XOR = 11,
		MATMUL = 12,
	};

	Register m_lhs;
	Register m_rhs;
	Operation m_operation;

  public:
	InplaceOp(Register lhs, Register rhs, Operation op) : m_lhs(lhs), m_rhs(rhs), m_operation(op) {}

	std::string to_string() const final
	{
		return fmt::format("INPLACE_OP      r{:<3} r{:<3} ({})", m_lhs, m_rhs, m_operation);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return INPLACE_OP; }
};
