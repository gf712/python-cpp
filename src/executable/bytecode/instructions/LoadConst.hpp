#pragma once

#include "Instructions.hpp"


class LoadConst final : public Instruction
{
	Register m_destination;
	size_t m_static_value_index;

  public:
	LoadConst(Register destination, size_t static_value_index)
		: m_destination(destination), m_static_value_index(static_value_index)
	{}
	~LoadConst() override {}
	std::string to_string() const final
	{
		return fmt::format("LOAD_CONST      r{:<3} s{:<3}", m_destination, m_static_value_index);
	}
	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final
	{
		ASSERT(vm.registers().has_value())
		ASSERT(vm.registers()->get().size() > m_destination)
		auto result = interpreter.execution_frame()->consts(m_static_value_index);
		vm.reg(m_destination) = result;
		return py::Ok(result);
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final
	{
		ASSERT(m_static_value_index < std::numeric_limits<uint8_t>::max())
		return {
			LOAD_CONST,
			m_destination,
			static_cast<u_int8_t>(m_static_value_index),
		};
	}
};
