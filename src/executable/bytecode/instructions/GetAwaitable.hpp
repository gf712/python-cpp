#pragma once

#include "Instructions.hpp"


class GetAwaitable final : public Instruction
{
	Register m_dst;
	Register m_src;

  public:
	GetAwaitable(Register dst, Register src) : m_dst(dst), m_src(src) {}

	std::string to_string() const final
	{
		return fmt::format("GET_AWAITABLE   r{:<3} r{:<3}", m_dst, m_src);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return GET_AWAITABLE; }
};
