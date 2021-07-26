#pragma once

#include "Instructions.hpp"

class LoadBuildClass final : public Instruction
{
	Register m_dst;

  public:
	LoadBuildClass(Register dst) : m_dst(dst) {}
	std::string to_string() const final { return fmt::format("LOAD_BUILD_CLASS r{}", m_dst); }

	void execute(VirtualMachine &vm, Interpreter &intepreter) const final
	{
		vm.reg(m_dst) = intepreter.fetch_object("__build_class__");
	}

	void relocate(BytecodeGenerator &, const std::vector<size_t> &) final {}
};
