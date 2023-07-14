#pragma once

#include "Instructions.hpp"

class FormatValue final : public Instruction
{
	const Register m_dst;
	const Register m_src;
	const uint8_t m_conversion;

  public:
	FormatValue(Register dst, Register src, uint8_t conversion)
		: m_dst(dst), m_src(src), m_conversion(conversion)
	{}

	std::string to_string() const final
	{
		return fmt::format("FORMAT_VALUE    r{:<3} r{:<3} {}", m_dst, m_src, m_conversion);
	}

	py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return FORMAT_VALUE; }
};
