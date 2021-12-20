#pragma once

#include "forward.hpp"
#include <memory>
#include <string>
#include <vector>

using InstructionVector = std::vector<std::unique_ptr<Instruction>>;

struct FunctionMetaData
{
	InstructionVector::const_iterator start_instruction;
	InstructionVector::const_iterator end_instruction;
	std::string function_name;
	size_t register_count{ 0 };
};

struct FunctionBlock
{
	FunctionMetaData metadata;
	InstructionVector instructions;
	std::string to_string() const;
};

using FunctionBlocks = std::vector<FunctionBlock>;
