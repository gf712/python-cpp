#pragma once

#include "Program.hpp"
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
	size_t stack_size{ 0 };
	std::vector<std::string> cellvars;
	std::vector<std::string> varnames;
	std::vector<std::string> freevars;
	std::vector<std::string> names;
	std::string filename;
	size_t first_line_number;
	size_t arg_count;
	size_t positional_arg_count;
	size_t kwonly_arg_count;
	size_t nlocals;
	std::vector<size_t> cell2arg;
	std::vector<py::Value> consts;
	CodeFlags flags = CodeFlags::create();
};

struct FunctionBlock
{
	FunctionMetaData metadata;
	std::list<InstructionBlock> blocks;
	std::string to_string() const;
};

struct FunctionBlocks
{
	std::list<FunctionBlock> functions;

	using FunctionType = decltype(functions)::value_type;
};
