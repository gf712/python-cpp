#pragma once

#include "forward.hpp"
#include "utilities.hpp"

#include <span>
#include <string>

class Instruction : NonCopyable
{
  public:
	virtual ~Instruction() = default;
	virtual std::string to_string() const = 0;
	virtual py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const = 0;
	virtual void relocate(size_t) = 0;
	virtual std::vector<uint8_t> serialize() const = 0;
	virtual uint8_t id() const = 0;
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
static constexpr uint8_t CONTINUE = 51;
static constexpr uint8_t RERAISE = 52;
static constexpr uint8_t WITH_EXCEPT_START = 53;
static constexpr uint8_t LEAVE_EXCEPTION_HANDLING = 54;
static constexpr uint8_t DELETE_SUBSCRIPT = 55;
static constexpr uint8_t SETUP_WITH = 56;
static constexpr uint8_t CLEAR_TOP_CLEANUP = 57;
static constexpr uint8_t LIST_APPEND = 58;
static constexpr uint8_t SET_ADD = 59;
static constexpr uint8_t BUILD_SET = 60;
static constexpr uint8_t IMPORT_FROM = 61;
static constexpr uint8_t BUILD_SLICE = 62;
static constexpr uint8_t INPLACE_OP = 63;
static constexpr uint8_t YIELD_VALUE = 64;
static constexpr uint8_t DICT_UPDATE = 65;
static constexpr uint8_t DICT_ADD = 66;
static constexpr uint8_t YIELD_LOAD = 67;
static constexpr uint8_t GET_YIELD_FROM_ITER = 68;
static constexpr uint8_t YIELD_FROM = 69;
static constexpr uint8_t IMPORT_STAR = 70;
static constexpr uint8_t GET_AWAITABLE = 71;
static constexpr uint8_t BUILD_STRING = 72;
static constexpr uint8_t FORMAT_VALUE = 73;
static constexpr uint8_t PUSH = 74;
static constexpr uint8_t POP = 75;
static constexpr uint8_t DELETE_FAST = 76;
static constexpr uint8_t DELETE_GLOBAL = 77;
static constexpr uint8_t JUMP_IF_EXCEPTION_MATCH = 78;
static constexpr uint8_t TO_BOOL = 79;
static constexpr uint8_t SET_UPDATE = 80;
static constexpr uint8_t UNPACK_EXPAND = 82;

std::unique_ptr<Instruction> deserialize(std::span<const uint8_t> &instruction_buffer);
