#pragma once

class RaiseVarargs final : public Instruction
{
	std::optional<Register> m_exception;
	std::optional<Register> m_cause;

  public:
	RaiseVarargs() {}
	RaiseVarargs(Register exception) : m_exception(exception) {}
	RaiseVarargs(Register exception, Register cause) : m_exception(exception), m_cause(cause) {}

	std::string to_string() const final
	{
		if (m_cause.has_value()) {
			ASSERT(m_exception.has_value());
			return std::format("RAISE_VARARGS   r{:<3} r{:<3}", *m_exception, *m_cause);
		} else if (m_exception.has_value()) {
			return std::format("RAISE_VARARGS   r{:<3}", *m_exception);
		} else {
			return std::format("RAISE_VARARGS");
		}
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(std::size_t) final {}

	std::vector<std::uint8_t> serialize() const final;

	std::uint8_t id() const final { return RAISE_VARARGS; }
};
