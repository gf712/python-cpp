#pragma once

#include "ast/AST.hpp"
#include "executable/Label.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"
#include "forward.hpp"
#include "interpreter/Interpreter.hpp"
#include "parser/Parser.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"

#include <sstream>

class Instruction : NonCopyable
{
  public:
	virtual ~Instruction() = default;
	virtual std::string to_string() const = 0;
	virtual void execute(VirtualMachine &, Interpreter &) const = 0;
	virtual void relocate(codegen::BytecodeGenerator &, size_t) = 0;
	// virtual std::vector<char> serialize() const = 0;
};

// std::vector<std::unique_ptr<Instruction>> deserialize(std::vector<char> instruction_stream);
