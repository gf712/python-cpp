#pragma once

#include "Instructions.hpp"


class StoreSubscript final : public Instruction
{
	Register m_obj;
	Register m_slice;
	Register m_src;

  public:
	StoreSubscript(Register obj, Register slice, Register source)
		: m_obj(obj), m_slice(slice), m_src(source)
	{}

	std::string to_string() const final
	{
		return fmt::format("STORE_SUBSCRIPT r{:<3} r{:<3} r{:<3}", m_obj, m_slice, m_src);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return STORE_SUBSCRIPT; }
};
