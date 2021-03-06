#pragma once

#include "ast/AST.hpp"
#include "executable/Label.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"
#include "forward.hpp"
#include "interpreter/Interpreter.hpp"
#include "parser/Parser.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"

#include <span>

class Instruction : NonCopyable
{
  public:
	virtual ~Instruction() = default;
	virtual std::string to_string() const = 0;
	virtual py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const = 0;
	virtual void relocate(codegen::BytecodeGenerator &, size_t) = 0;
	virtual std::vector<uint8_t> serialize() const = 0;
};

static constexpr uint8_t BINARY_OPERATION = 0;
static constexpr uint8_t BINARY_SUBSCRIPT = 1;
static constexpr uint8_t BUILD_DICT = 2;
static constexpr uint8_t BUILD_LIST = 3;
static constexpr uint8_t BUILD_TUPLE = 4;
static constexpr uint8_t CLEAR_EXCEPTION_STATE = 5;
static constexpr uint8_t COMPARE_OP = 6;
static constexpr uint8_t DELETE_NAME = 7;
static constexpr uint8_t DICT_MERGE = 8;
static constexpr uint8_t FOR_ITER = 9;
static constexpr uint8_t FUNCTION_CALL = 10;
static constexpr uint8_t FUNCTION_CALL_EX = 11;
static constexpr uint8_t FUNCTION_CALL_WITH_KW = 12;
static constexpr uint8_t GET_ITER = 13;
static constexpr uint8_t IMPORT_NAME = 14;
static constexpr uint8_t INPLACE_ADD = 15;
static constexpr uint8_t INPLACE_SUB = 16;
static constexpr uint8_t JUMP = 17;
static constexpr uint8_t JUMP_FORWARD = 18;
static constexpr uint8_t JUMP_IF_FALSE = 19;
static constexpr uint8_t JUMP_IF_FALSE_OR_POP = 20;
static constexpr uint8_t JUMP_IF_NOT_EXCEPTION_MATCH = 21;
static constexpr uint8_t JUMP_IF_TRUE = 22;
static constexpr uint8_t JUMP_IF_TRUE_OR_POP = 23;
static constexpr uint8_t LIST_EXTEND = 24;
static constexpr uint8_t LIST_TO_TUPLE = 25;
static constexpr uint8_t LOAD_ASSERTION_ERROR = 26;
static constexpr uint8_t LOAD_ATTR = 27;
static constexpr uint8_t LOAD_BUILD_CLASS = 28;
static constexpr uint8_t LOAD_CLOSURE = 29;
static constexpr uint8_t LOAD_CONST = 30;
static constexpr uint8_t LOAD_DEREF = 31;
static constexpr uint8_t LOAD_FAST = 32;
static constexpr uint8_t LOAD_GLOBAL = 33;
static constexpr uint8_t LOAD_METHOD = 34;
static constexpr uint8_t LOAD_NAME = 35;
static constexpr uint8_t MAKE_FUNCTION = 36;
static constexpr uint8_t MERGE_DICT = 37;
static constexpr uint8_t METHOD_CALL = 38;
static constexpr uint8_t MOVE = 39;
static constexpr uint8_t RAISE_VARARGS = 40;
static constexpr uint8_t RETURN_VALUE = 41;
static constexpr uint8_t SETUP_EXCEPTION_HANDLING = 42;
static constexpr uint8_t STORE_ATTR = 43;
static constexpr uint8_t STORE_DEREF = 44;
static constexpr uint8_t STORE_FAST = 45;
static constexpr uint8_t STORE_GLOBAL = 46;
static constexpr uint8_t STORE_NAME = 47;
static constexpr uint8_t STORE_SUBSCRIPT = 48;
static constexpr uint8_t UNARY = 49;
static constexpr uint8_t UNPACK_SEQUENCE = 50;

std::unique_ptr<Instruction> deserialize(std::span<const uint8_t> &instruction_buffer);
