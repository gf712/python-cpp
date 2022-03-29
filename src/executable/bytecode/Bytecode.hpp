#pragma once

#include "codegen/BytecodeGenerator.hpp"
#include "executable/Function.hpp"
#include "executable/FunctionBlock.hpp"

#include "forward.hpp"

#include <memory>
#include <set>
#include <string>
#include <vector>
#include <span>

class Bytecode : public Function
{
	const InstructionVector m_instructions;
	const std::vector<View> m_block_views;

  public:
	Bytecode(size_t register_count,
		size_t stack_size,
		std::string function_name,
		InstructionVector &&instructions,
		std::vector<View> block_views);

	auto begin() const { return m_block_views.begin(); }
	auto end() const { return m_block_views.end(); }

	std::string to_string() const override;

	std::vector<uint8_t> serialize() const override;

	static std::unique_ptr<Bytecode> deserialize(std::span<const uint8_t> &buffer);
};
