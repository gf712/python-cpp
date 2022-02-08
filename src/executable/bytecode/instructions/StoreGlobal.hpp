#pragma once

#include "Instructions.hpp"


class StoreGlobal final : public Instruction
{
	std::string m_object_name;
	Register m_source;

  public:
	StoreGlobal(std::string object_name, Register source)
		: m_object_name(std::move(object_name)), m_source(source)
	{}
	std::string to_string() const final
	{
		return fmt::format("STORE_GLOBAL    \"{}\" r{:<3}", m_object_name, m_source);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};