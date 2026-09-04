#pragma once

class SetupWith final : public Instruction
{
	std::shared_ptr<Label> m_label;
	std::optional<std::uint32_t> m_offset;

  public:
	SetupWith(std::shared_ptr<Label> label) : m_label(std::move(label)) {}

	SetupWith(std::uint32_t offset) : m_offset(offset) {}

	std::string to_string() const final
	{
		const std::string position =
			m_offset.has_value() ? std::to_string(*m_offset) : "offset not evaluated";
		return std::format("SETUP_WITH      position: {}", position);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(std::size_t) final;

	std::vector<std::uint8_t> serialize() const final;

	std::uint8_t id() const final { return SETUP_WITH; }
};
