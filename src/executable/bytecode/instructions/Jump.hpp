#pragma once

#include "Instructions.hpp"


class Jump final : public Instruction
{
	std::shared_ptr<Label> m_label;
	std::optional<int32_t> m_offset;

  public:
	Jump(std::shared_ptr<Label> label) : m_label(std::move(label)) {}
	std::string to_string() const final
	{
		return fmt::format("JUMP            position: {}", m_label->position());
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;
};
