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
	std::vector<View> m_block_views;

  public:
	Bytecode(size_t registers_needed, std::string m_function_name, std::vector<View> block_views);

	auto begin() const { return m_block_views.begin(); }
	auto end() const { return m_block_views.end(); }

	std::string to_string() const override;
};
