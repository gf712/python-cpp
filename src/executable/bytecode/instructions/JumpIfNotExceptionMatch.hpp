#include "Instructions.hpp"


class JumpIfNotExceptionMatch final : public Instruction
{
	Register m_exception_type_reg;

  public:
	JumpIfNotExceptionMatch(Register exception_type_reg) : m_exception_type_reg(exception_type_reg)
	{}

	std::string to_string() const final
	{
		return fmt::format("JUMP_IF_NOT_EXC r{}", m_exception_type_reg);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};
