#pragma once

#include "Instructions.hpp"

class MethodCall final : public Instruction
{
	const Register m_caller;
	const std::string m_function_name;
	const std::vector<Register> m_args;

  public:
	MethodCall(Register caller, std::string function_name, std::vector<Register> &&args)
		: m_caller(caller), m_function_name(std::move(function_name)), m_args(std::move(args))
	{}

	std::string to_string() const final
	{
		std::string args_regs{};
		for (const auto arg : m_args) { args_regs += fmt::format(" r{:<3}", arg); }
		return fmt::format("CALL_METHOD     r{:<3} {} ()", m_caller, args_regs, m_function_name);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
