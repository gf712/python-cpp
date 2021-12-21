#pragma once

#include "forward.hpp"
#include <list>
#include <memory>
#include <string>
#include <vector>


using InstructionVector = std::vector<std::unique_ptr<Instruction>>;
using InstructionBlock = InstructionVector;

struct FunctionMetaData
{
	InstructionBlock::const_iterator start_instruction;
	InstructionBlock::const_iterator end_instruction;
	std::string function_name;
	size_t register_count{ 0 };
};

struct FunctionBlock
{
	FunctionMetaData metadata;
	std::vector<InstructionBlock> blocks;
	std::string to_string() const;
};

using FunctionBlocks = std::list<FunctionBlock>;
