#pragma once

#include "Instructions.hpp"

#include <optional>

class SetupExceptionHandling final : public Instruction
{
	std::shared_ptr<Label> m_label;
	std::optional<uint32_t> m_offset;

  public:
	SetupExceptionHandling(std::shared_ptr<Label> label) : m_label(std::move(label)) {}

	SetupExceptionHandling(uint32_t offset) : m_offset(offset) {}

	std::string to_string() const final
	{
		const std::string position =
			m_offset.has_value() ? std::to_string(*m_offset) : "offset not evaluated";
		return fmt::format("SETUP_EXC_HANDLE position: {}", position);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return SETUP_EXCEPTION_HANDLING; }
};
