#include "Instructions.hpp"

class JumpForward final : public Instruction
{
	size_t m_block_count;

  public:
	JumpForward(size_t block_count) : m_block_count(block_count) {}

	std::string to_string() const final { return fmt::format("JUMP_FORWARD    {}", m_block_count); }

	py::PyResult execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};
