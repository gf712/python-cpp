#pragma once

#include "Instructions.hpp"


class StoreDeref final : public Instruction
{
	Register m_dst;
	Register m_src;

  public:
	StoreDeref(Register dst, Register src) : m_dst(dst), m_src(src) {}

	std::string to_string() const final
	{
		return fmt::format("STORE_DEREF     f{:<3} r{:<3}", m_dst, m_src);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return STORE_DEREF; }
};
