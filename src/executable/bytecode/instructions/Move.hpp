#pragma once

#include "Instructions.hpp"

class Move final : public Instruction
{
	Register m_destination;
	Register m_source;

  public:
	Move(Register destination, Register source) : m_destination(destination), m_source(source) {}
	~Move() override {}
	std::string to_string() const final
	{
		return fmt::format("MOVE            r{:<3}  r{:<3}", m_destination, m_source);
	}
	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};