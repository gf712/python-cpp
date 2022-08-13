#pragma once

#include "Instructions.hpp"

class BuildSlice final : public Instruction
{
	Register m_dst;
	std::optional<size_t> m_start;
	std::optional<size_t> m_end;
	std::optional<size_t> m_step;


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
			return fmt::format("BUILD_SLICE r{:<3} r{:<3}", m_dst, *m_start);
		} else if (!m_step) {
			return fmt::format("BUILD_SLICE r{:<3} r{:<3} r{:<3}", m_dst, *m_start, *m_end);
		} else {
			return fmt::format(
				"BUILD_SLICE r{:<3} r{:<3} r{:<3} r{:<3}", m_dst, *m_start, *m_end, *m_step);
		}
	}

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return BUILD_SET; }
};
