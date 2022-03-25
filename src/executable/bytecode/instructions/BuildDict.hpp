#pragma once

#include "Instructions.hpp"


class BuildDict final : public Instruction
{
	Register m_dst;
	std::vector<Register> m_keys;
	std::vector<Register> m_values;

  public:
	BuildDict(Register dst, std::vector<Register> keys, std::vector<Register> values)
		: m_dst(dst), m_keys(std::move(keys)), m_values(std::move(values))
	{}

	std::string to_string() const final { return fmt::format("BUILD_DICT     r{:<3}", m_dst); }

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};