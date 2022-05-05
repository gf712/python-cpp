#pragma once

#include "Instructions.hpp"


class JumpIfFalse final : public Instruction
{
	Register m_test_register;
	std::shared_ptr<Label> m_label;
	std::optional<int32_t> m_offset;

  public:
	JumpIfFalse(Register test_register, std::shared_ptr<Label> label)
		: m_test_register(test_register), m_label(std::move(label))
	{}

	std::string to_string() const final
	{
		return fmt::format("JUMP_IF_FALSE   position: {}", m_label->position());
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;

	std::vector<uint8_t> serialize() const final;
};
