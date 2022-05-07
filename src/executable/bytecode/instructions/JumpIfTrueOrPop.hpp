#include "Instructions.hpp"


class JumpIfTrueOrPop final : public Instruction
{
	Register m_test_register;
	Register m_result_register;
	std::shared_ptr<Label> m_label;
	std::optional<int32_t> m_offset;

  public:
	JumpIfTrueOrPop(Register test_register, Register result_register, std::shared_ptr<Label> label)
		: m_test_register(test_register), m_result_register(result_register),
		  m_label(std::move(label))
	{}

	std::string to_string() const final
	{
		return fmt::format("JUMP_IF_TRUE_OR_POP position: {}", m_label->position());
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return JUMP_IF_TRUE_OR_POP; }
};
