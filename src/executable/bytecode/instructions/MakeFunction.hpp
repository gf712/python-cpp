#pragma once

#include "Instructions.hpp"


class MakeFunction : public Instruction
{
	Register m_dst;
	std::string m_function_name;
	std::vector<Register> m_defaults;
	std::vector<std::optional<Register>> m_kw_defaults;
	std::optional<Register> m_captures_tuple;

  public:
	MakeFunction(Register dst,
		std::string function_name,
		std::vector<Register> defaults,
		std::vector<std::optional<Register>> kw_defaults,
		std::optional<Register> captures_tuple)
		: m_dst(dst), m_function_name(std::move(function_name)), m_defaults(std::move(defaults)),
		  m_kw_defaults(std::move(kw_defaults)), m_captures_tuple(std::move(captures_tuple))
	{}

	std::string to_string() const final
	{
		return fmt::format("MAKE_FUNCTION   ({})", m_function_name);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};
