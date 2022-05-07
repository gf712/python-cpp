#pragma once

#include "Instructions.hpp"


class StoreName final : public Instruction
{
	std::string m_object_name;
	Register m_source;

  public:
	StoreName(std::string object_name, Register source)
		: m_object_name(std::move(object_name)), m_source(source)
	{}
	std::string to_string() const final
	{
		return fmt::format("STORE_NAME      \"{}\" r{:<3}", m_object_name, m_source);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return STORE_NAME; }
};