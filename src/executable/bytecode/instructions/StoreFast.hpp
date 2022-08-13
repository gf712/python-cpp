#pragma once

#include "Instructions.hpp"

class StoreFast final : public Instruction
{
	Register m_stack_index;
	Register m_src;

  public:
	StoreFast(size_t stack_index, Register src) : m_stack_index(stack_index), m_src(src) {}
	~StoreFast() override {}
	std::string to_string() const final
	{
		return fmt::format("STORE_FAST       {} r{:<3}", m_stack_index, m_src);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const override;

	uint8_t id() const final { return STORE_FAST; }
};
