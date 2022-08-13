#pragma once

#include "Instructions.hpp"

class FunctionCallWithKeywords final : public Instruction
{
	Register m_function_name;
	std::vector<Register> m_args;
	std::vector<Register> m_kwargs;
	std::vector<Register> m_keywords;

  public:
	FunctionCallWithKeywords(Register function_name,
		std::vector<Register> &&args,
		std::vector<Register> &&kwargs,
		std::vector<Register> &&keywords)
		: m_function_name(function_name), m_args(std::move(args)),
		  m_kwargs(std::move(kwargs)), m_keywords(std::move(keywords))
	{}
	std::string to_string() const final
	{
		std::string args_regs{};
		std::string kwargs_regs{};
		std::string keyword_regs{};

		for (const auto arg : m_args) { args_regs += fmt::format(" r{:<3}", arg); }
		for (const auto kwarg : m_kwargs) { kwargs_regs += fmt::format(" r{:<3}", kwarg); }
		for (const auto &keyword : m_keywords) { keyword_regs += fmt::format("r{:<3} ", keyword); }

		return fmt::format("CALL_FUNCTION_KW r{:<3}args={} kwargs={} keywords={}",
			m_function_name,
			args_regs,
			kwargs_regs,
			keyword_regs);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;

	uint8_t id() const final { return FUNCTION_CALL_WITH_KW; }
};
