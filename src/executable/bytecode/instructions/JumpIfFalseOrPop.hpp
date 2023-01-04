#pragma once
#include "Instructions.hpp"

#include <optional>

class JumpIfFalseOrPop final : public Instruction
{
	Register m_test_register;
	Register m_result_register;
	std::shared_ptr<Label> m_label;
	std::optional<int32_t> m_offset;

  public:
	JumpIfFalseOrPop(Register test_register, Register result_register, std::shared_ptr<Label> label)
		: m_test_register(test_register), m_result_register(result_register),
		  m_label(std::move(label))
	{}

	JumpIfFalseOrPop(Register test_register, Register result_register, int32_t offset)
		: m_test_register(test_register), m_result_register(result_register), m_offset(offset)
	{}

	std::string to_string() const final
	{
		const std::string position =
			m_offset.has_value() ? std::to_string(*m_offset) : "offset not evaluated";
		return fmt::format("JUMP_IF_FALSE_OR_POP r{:<3} r{:<3} position: {}",
			m_test_register,
			m_result_register,
			position);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final;

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return JUMP_IF_FALSE_OR_POP; }
};
