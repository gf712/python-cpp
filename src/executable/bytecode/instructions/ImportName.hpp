#pragma once

#include "Instructions.hpp"


class ImportName : public Instruction
{
	Register m_destination;
	std::vector<std::string> m_names;

  public:
	ImportName(Register dst, std::vector<std::string> name)
		: m_destination(dst), m_names(std::move(name))
	{}

	std::string to_string() const final
	{
		std::string name = std::accumulate(
			std::next(m_names.begin()), m_names.end(), *m_names.begin(), [](auto rhs, auto lhs) {
				return std::move(rhs) + "." + lhs;
			});
		return fmt::format("IMPORT_NAME     r{:<3} {:<3}", m_destination, name);
	}

	py::PyResult<py::Value> execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}

	std::vector<uint8_t> serialize() const final;
};