#pragma once

#include "Instructions.hpp"


class ReturnValue final : public Instruction
{
	Register m_source;

  public:
	ReturnValue(Register source) : m_source(source) {}
	~ReturnValue() override {}
	std::string to_string() const final { return fmt::format("RETURN_VALUE    r{:<3}", m_source); }

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;
	void relocate(BytecodeGenerator &, const std::vector<size_t> &) final {}
};