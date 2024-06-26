#pragma once

#include "codegen/BytecodeGenerator.hpp"
#include "executable/Function.hpp"
#include "executable/FunctionBlock.hpp"

#include "forward.hpp"

#include <memory>
#include <set>
#include <span>
#include <string>
#include <vector>

class Bytecode : public Function
{
	const InstructionVector m_instructions;

  public:
	Bytecode(size_t register_count,
		size_t locals_count,
		size_t stack_size,
		std::string function_name,
		InstructionVector instructions,
		std::shared_ptr<Program> program);

	auto begin() const { return m_instructions.begin(); }
	auto end() const { return m_instructions.end(); }

	std::string to_string() const override;

	std::vector<uint8_t> serialize() const override;

	static std::unique_ptr<Bytecode> deserialize(std::span<const uint8_t> &buffer, std::shared_ptr<Program> program);

	py::PyResult<py::Value> call(VirtualMachine &, Interpreter &) const override;
	py::PyResult<py::Value> call_without_setup(VirtualMachine &, Interpreter &) const override;

	py::PyResult<py::Value> eval_loop(VirtualMachine &, Interpreter &) const;
};
