#pragma once

#include "Instructions.hpp"


class Unary final : public Instruction
{
  public:
	enum class Operation {
		POSITIVE = 0,
		NEGATIVE = 1,
		INVERT = 2,
		NOT = 3,
	};

	Register m_destination;
	Register m_source;
	Operation m_operation;

  public:
	Unary(Register destination, Register source, Operation operation)
		: m_destination(destination), m_source(source), m_operation(operation)
	{}

	std::string to_string() const final
	{
		std::array op_str{ "+", "-", "~", "!" };
		return fmt::format("UNARY  r{:<3} r{:<3} ({})",
			m_destination,
			m_source,
			op_str[static_cast<uint8_t>(m_operation)]);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return UNARY; }
};
