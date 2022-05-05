#pragma once

#include "Instructions.hpp"


class CompareOperation final : public Instruction
{
  public:
	enum class Comparisson {
		Eq = 0,
		NotEq = 1,
		Lt = 2,
		LtE = 3,
		Gt = 4,
		GtE = 5,
		Is = 6,
		IsNot = 7,
		In = 8,
		NotIn = 9,
	};

  private:
	Register m_dst;
	Register m_lhs;
	Register m_rhs;
	Comparisson m_comparisson;

  public:
	CompareOperation(Register dst, Register lhs, Register rhs, Comparisson comparisson)
		: m_dst(dst), m_lhs(lhs), m_rhs(rhs), m_comparisson(comparisson)
	{}

	std::string to_string() const final
	{
		return fmt::format(
			"COMPARE_OP      r{:<3} r{:<3} r{:<3} ({})", m_dst, m_lhs, m_rhs, m_comparisson);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};