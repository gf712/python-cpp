#pragma once

#include "Instructions.hpp"

#include <optional>

class JumpIfFalse final : public Instruction
{
	Register m_test_register;
	std::shared_ptr<Label> m_label;
	std::optional<int32_t> m_offset;

  public:
	JumpIfFalse(Register test_register, std::shared_ptr<Label> label)
		: m_test_register(test_register), m_label(std::move(label))
	{}

	JumpIfFalse(Register test_register, int32_t offset)
		: m_test_register(test_register), m_offset(offset)
	{}

	std::string to_string() const final
	{
		const std::string position =
			m_offset.has_value() ? std::to_string(*m_offset) : "offset not evaluated";
		return fmt::format("JUMP_IF_FALSE   r{:<3} position: {}", m_test_register, position);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final;

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return JUMP_IF_FALSE; }
};
