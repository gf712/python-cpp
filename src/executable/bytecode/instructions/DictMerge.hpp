#pragma once

#include "Instructions.hpp"


class DictMerge final : public Instruction
{
	Register m_this_dict;
	Register m_other_dict;

  public:
	DictMerge(Register this_dict, Register other_dict)
		: m_this_dict(this_dict), m_other_dict(other_dict)
	{}

	std::string to_string() const final
	{
		return fmt::format("DICT_MERGE      r{:<3} r{:<3}", m_this_dict, m_other_dict);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return DICT_MERGE; }
};
