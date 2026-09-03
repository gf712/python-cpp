module;

#include "CodeFlags.hpp"

export module py.runtime:executable_functionblock;
import :value;
import std;

export class Instruction;


export using InstructionVector = std::vector<std::unique_ptr<Instruction>>;

export struct InstructionSourceLocation
{
	std::uint32_t instruction_index;
	std::uint32_t line;
	std::uint32_t column;
};

export struct FunctionMetaData
{
	std::string function_name;
	std::size_t register_count{ 0 };
	std::size_t stack_size{ 0 };
	std::vector<std::string> cellvars;
	std::vector<std::string> varnames;
	std::vector<std::string> freevars;
	std::vector<std::string> names;
	std::string filename;
	std::size_t first_line_number;
	std::size_t arg_count;
	std::size_t positional_arg_count;
	std::size_t kwonly_arg_count;
	std::size_t nlocals;
	std::vector<std::size_t> cell2arg;
	std::vector<py::Value> consts;
	CodeFlags flags = CodeFlags::create();
};

export struct FunctionBlock
{
	FunctionMetaData metadata;
	InstructionVector blocks;
	std::vector<InstructionSourceLocation> instruction_locations;
	std::string to_string() const;
};

export struct FunctionBlocks
{
	std::list<FunctionBlock> functions;

	using FunctionType = decltype(functions)::value_type;
};
