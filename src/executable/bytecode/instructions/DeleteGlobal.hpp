#pragma once

class DeleteGlobal final : public Instruction
{
	Register m_name;

  public:
	DeleteGlobal(Register name) : m_name(name) {}

	std::string to_string() const final { return std::format("DELETE_GLOBAL   r{:<3}", m_name); }

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return DELETE_GLOBAL; }
};
