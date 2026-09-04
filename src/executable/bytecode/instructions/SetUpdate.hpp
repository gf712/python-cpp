#pragma once


class SetUpdate final : public Instruction
{
	Register m_set;
	Register m_iterable;

  public:
	SetUpdate(Register set, Register iterable) : m_set(set), m_iterable(iterable) {}

	std::string to_string() const final
	{
		return std::format("SET_UPDATE      r{:<3} r{:<3}", m_set, m_iterable);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return SET_UPDATE; }
};
