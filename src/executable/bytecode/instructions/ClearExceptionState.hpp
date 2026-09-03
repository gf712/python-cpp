#pragma once

class ClearExceptionState final : public Instruction
{
  public:
	ClearExceptionState() = default;
	std::string to_string() const final { return std::format("CLEAR_EXC"); }

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return CLEAR_EXCEPTION_STATE; }
};
