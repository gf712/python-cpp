#pragma once

#include "Instructions.hpp"


class DictAdd final : public Instruction
{
	Register m_dict;
	Register m_key;
	Register m_value;

  public:
	DictAdd(Register dict, Register key, Register value) : m_dict(dict), m_key(key), m_value(value)
	{}

	std::string to_string() const final
	{
		return fmt::format("DICT_ADD        r{:<3} r{:<3} r{:<3}", m_dict, m_key, m_value);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return DICT_ADD; }
};
