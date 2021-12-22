#include "Instructions.hpp"

class SetupExceptionHandling final : public Instruction
{
  public:
	SetupExceptionHandling() = default;

	std::string to_string() const final { return fmt::format("SETUP_EXC_HANDLE"); }

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
