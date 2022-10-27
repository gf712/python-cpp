#pragma once

#include "Instructions.hpp"

#include <optional>

class Jump final : public Instruction
{
	std::shared_ptr<Label> m_label;
	std::optional<int32_t> m_offset;

  public:
	Jump(std::shared_ptr<Label> label) : m_label(std::move(label)) {}

	Jump(int32_t offset) : m_offset(offset) {}

	std::string to_string() const final
	{
		const std::string position =
			m_offset.has_value() ? std::to_string(*m_offset) : "offset not evaluated";
		return fmt::format("JUMP            position: {}", position);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return JUMP; }
};
