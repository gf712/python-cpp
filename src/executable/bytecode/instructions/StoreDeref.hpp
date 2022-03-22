#pragma once

#include "Instructions.hpp"


class StoreDeref final : public Instruction
{
	Register m_dst;
	Register m_src;

  public:
	StoreDeref(Register dst, Register src) : m_dst(dst), m_src(src) {}

	std::string to_string() const final
	{
		return fmt::format("STORE_DEREF     f{:<3} r{:<3}", m_dst, m_src);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};