#pragma once

#include "Instructions.hpp"

class MethodCall final : public Instruction
{
	const Register m_caller;
	const std::vector<Register> m_args;

  public:
	MethodCall(Register caller, std::vector<Register> &&args)
		: m_caller(caller), m_args(std::move(args))
	{}

	std::string to_string() const final
	{
		std::string args_regs{};
		for (const auto arg : m_args) { args_regs += fmt::format(" r{:<3}", arg); }
		return fmt::format("CALL_METHOD     r{:<3} {}", m_caller, args_regs);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return METHOD_CALL; }
};
