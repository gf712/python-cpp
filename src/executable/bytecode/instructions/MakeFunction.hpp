#pragma once

class MakeFunction : public Instruction
{
	Register m_dst;
	Register m_name;
	std::size_t m_defaults_size;
	std::size_t m_kw_defaults_size;
	std::optional<Register> m_captures_tuple;

  public:
	MakeFunction(Register dst,
		Register function_name,
		std::size_t defaults_size,
		std::size_t kw_defaults_size,
		std::optional<Register> captures_tuple)
		: m_dst(dst), m_name(function_name), m_defaults_size(defaults_size),
		  m_kw_defaults_size(kw_defaults_size), m_captures_tuple(std::move(captures_tuple))
	{}

	std::string to_string() const final
	{
		return std::format("MAKE_FUNCTION   r{}   ({})", m_dst, m_name);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(std::size_t) final {}

	std::vector<std::uint8_t> serialize() const final;

	std::uint8_t id() const final { return MAKE_FUNCTION; }
};
