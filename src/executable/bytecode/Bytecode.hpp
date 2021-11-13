#pragma once

#include "codegen/BytecodeGenerator.hpp"
#include "executable/Function.hpp"
#include "executable/FunctionBlock.hpp"

#include "forward.hpp"

#include <memory>
#include <set>
#include <string>
#include <vector>

class Bytecode : public Function
{
	View m_bytecode_view;

  public:
	Bytecode(size_t registers_needed,
		std::string m_function_name,
		InstructionVector::const_iterator begin,
		InstructionVector::const_iterator end);

	auto begin() const { return m_bytecode_view.begin(); }
	auto end() const { return m_bytecode_view.end(); }

	std::string to_string() const override;
};
