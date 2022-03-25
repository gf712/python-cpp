#pragma once

#include "Instructions.hpp"

class LoadFast final : public Instruction
{
	Register m_destination;
	Register m_stack_index;
	const std::string m_object_name;

  public:
	LoadFast(Register destination, Register stack_index, std::string object_name)
		: m_destination(destination), m_stack_index(stack_index),
		  m_object_name(std::move(object_name))
	{}
	~LoadFast() override {}
	std::string to_string() const final
	{
		return fmt::format(
			"LOAD_FAST       r{:<3} {} (\"{}\")", m_destination, m_stack_index, m_object_name);
	}

	Register dst() const { return m_destination; }
	Register stack_index() const { return m_stack_index; }
	const std::string &object_name() const { return m_object_name; }

	void execute(VirtualMachine &vm, Interpreter &) const final
	{
		vm.reg(m_destination) = vm.stack_local(m_stack_index);
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
