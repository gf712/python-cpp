#pragma once

#include "Instructions.hpp"


class LoadClosure final : public Instruction
{
	Register m_destination;
	Register m_source;
	std::string m_object_name;

  public:
	LoadClosure(Register destination, Register source, std::string object_name)
		: m_destination(destination), m_source(source), m_object_name(std::move(object_name))
	{}

	std::string to_string() const final
	{
		return fmt::format(
			"LOAD_CLOSURE    r{:<3} f{:<3} \"{}\"", m_destination, m_source, m_object_name);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};