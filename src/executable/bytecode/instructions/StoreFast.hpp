#pragma once

#include "Instructions.hpp"

class StoreFast final : public Instruction
{
	Register m_stack_index;
	const std::string m_object_name;
	Register m_src;

  public:
	StoreFast(size_t stack_index, std::string object_name, Register src)
		: m_stack_index(stack_index), m_object_name(std::move(object_name)), m_src(src)
	{}
	~StoreFast() override {}
	std::string to_string() const final
	{
		return fmt::format(
			"STORE_FAST       {} (\"{}\") r{:<3}", m_stack_index, m_object_name, m_src);
	}

	void execute(VirtualMachine &vm, Interpreter &) const final
	{
		vm.stack_local(m_stack_index) = vm.reg(m_src);
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
