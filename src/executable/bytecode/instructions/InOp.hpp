#include "Instructions.hpp"

class InOp final : public Instruction
{
	Register m_dst;
	Register m_lhs;
	Register m_rhs;
	bool m_not_in;

  public:
	InOp(Register dst, Register lhs, Register rhs, bool not_in)
		: m_dst(dst), m_lhs(lhs), m_rhs(rhs), m_not_in(not_in)
	{}

	std::string to_string() const final
	{
		return fmt::format(
			"IN_OP           r{:<3} r{:<3} r{:<3} ({})", m_dst, m_lhs, m_rhs, m_not_in);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
