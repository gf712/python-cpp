#pragma once

#include "Instructions.hpp"


class BuildList final : public Instruction
{
	Register m_dst;
	std::vector<Register> m_srcs;

  public:
	BuildList(Register dst, std::vector<Register> srcs) : m_dst(dst), m_srcs(std::move(srcs)) {}

	std::string to_string() const final { return fmt::format("BUILD_LIST      r{:<3}", m_dst); }

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
