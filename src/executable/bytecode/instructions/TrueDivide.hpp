#include "Instructions.hpp"

class TrueDivide final : public Instruction
{
	Register m_dst;
	Register m_lhs;
	Register m_rhs;

  public:
	TrueDivide(Register dst, Register lhs, Register rhs) : m_dst(dst), m_lhs(lhs), m_rhs(rhs) {}

	std::string to_string() const final
	{
		return fmt::format("TRUE_DIVIDE     r{:<3} r{:<3} r{:<3}", m_dst, m_lhs, m_rhs);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};