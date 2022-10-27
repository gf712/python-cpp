#pragma once

#include "Instructions.hpp"

#include <optional>

class SetupWith final : public Instruction
{
	std::shared_ptr<Label> m_label;
	std::optional<uint32_t> m_offset;

  public:
	SetupWith(std::shared_ptr<Label> label) : m_label(std::move(label)) {}

	SetupWith(uint32_t offset) : m_offset(offset) {}

	std::string to_string() const final
	{
		const std::string position =
			m_offset.has_value() ? std::to_string(*m_offset) : "offset not evaluated";
		return fmt::format("SETUP_WITH      position: {}", position);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return SETUP_WITH; }
};
