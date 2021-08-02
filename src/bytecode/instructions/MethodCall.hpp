#pragma once

#include "Instructions.hpp"

class MethodCall final : public Instruction
{
	Register m_function_name;
	std::vector<Register> m_args;

  public:
	MethodCall(Register function_name, std::vector<Register> &&args)
		: m_function_name(std::move(function_name)), m_args(std::move(args))
	{}
	std::string to_string() const final
	{
		std::string args_regs{};
		for (const auto arg : m_args) { args_regs += fmt::format(" r{:<3}", arg); }
		return fmt::format("CALL_METHOD     r{:<3}{}", m_function_name, args_regs);
	}

	void execute(VirtualMachine &, Interpreter &) const final {TODO()}

	void relocate(BytecodeGenerator &, const std::vector<size_t> &) final {}
};
