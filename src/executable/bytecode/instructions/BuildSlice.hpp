#pragma once

class BuildSlice final : public Instruction
{
	Register m_dst;
	std::optional<std::size_t> m_start;
	std::optional<std::size_t> m_end;
	std::optional<std::size_t> m_step;


  public:
	BuildSlice(Register dst, Register start) : m_dst(dst), m_start(start) {}

	BuildSlice(Register dst, Register start, Register end) : m_dst(dst), m_start(start), m_end(end)
	{}

	BuildSlice(Register dst, Register start, Register end, Register step)
		: m_dst(dst), m_start(start), m_end(end), m_step(step)
	{}

	std::string to_string() const final
	{
		if (!m_end) {
			return std::format("BUILD_SLICE r{:<3} r{:<3}", m_dst, *m_start);
		} else if (!m_step) {
			return std::format("BUILD_SLICE r{:<3} r{:<3} r{:<3}", m_dst, *m_start, *m_end);
		} else {
			return std::format(
				"BUILD_SLICE r{:<3} r{:<3} r{:<3} r{:<3}", m_dst, *m_start, *m_end, *m_step);
		}
	}

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(std::size_t) final {}

	std::vector<std::uint8_t> serialize() const final;

	std::uint8_t id() const final { return BUILD_SET; }
};
