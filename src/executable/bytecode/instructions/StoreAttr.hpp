#pragma once

#include "Instructions.hpp"


class StoreAttr final : public Instruction
{
	Register m_dst;
	Register m_src;
	const std::string m_attr_name;

  public:
	StoreAttr(Register destination, Register value_source, std::string attr_name)
		: m_dst(destination), m_src(value_source), m_attr_name(std::move(attr_name))
	{}

	std::string to_string() const final
	{
		return fmt::format("STORE_ATTR      r{:<3} ({}) r{:<3}", m_dst, m_attr_name, m_src);
	}

	py::PyResult execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};