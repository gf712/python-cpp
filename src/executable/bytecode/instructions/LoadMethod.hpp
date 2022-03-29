#pragma once

#include "Instructions.hpp"


class LoadMethod final : public Instruction
{
	Register m_destination;
	Register m_value_source;
	const std::string m_method_name;

  public:
	LoadMethod(Register destination, Register value_source, std::string method_name)
		: m_destination(destination), m_value_source(value_source), m_method_name(std::move(method_name))
	{}

	std::string to_string() const final
	{
		return fmt::format("LOAD_METHOD     r{:<3} r{:<3} ({})", m_destination, m_value_source, m_method_name);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};