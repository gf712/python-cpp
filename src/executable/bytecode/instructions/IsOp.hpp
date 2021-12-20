#include "Instructions.hpp"

class IsOp final : public Instruction
{
	Register m_dst;
	Register m_lhs;
	Register m_rhs;
	bool m_is_not;

  public:
	IsOp(Register dst, Register lhs, Register rhs, bool is_not)
		: m_dst(dst), m_lhs(lhs), m_rhs(rhs), m_is_not(is_not)
	{}

	std::string to_string() const final
	{
		return fmt::format("IS_OP           r{:<3} r{:<3} r{:<3} ()", m_dst, m_lhs, m_rhs);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
