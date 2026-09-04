module;

#include "core.hpp"
#include "executable/Label.hpp"

export module py.runtime:instructions;
import :value;
import std;

export class Program;

// Instruction is referenced from both py.runtime (FunctionBlock's
// InstructionVector) and py.codegen, so it has to have exactly one owning
// module - as a plain header included from two different purviews it was being
// attached to whichever module included it.
export class VirtualMachine;
export class Interpreter;

export class Instruction : NonCopyable
{
  protected:
	// RAII type that restores r0 for intructions that are not treated as call instructions
	// but may end up calling a Python function (e.g. LoadAttribute may end up calling a Python
	// defined __getattr__ which then clobbers r0)
	struct RAIIStoreNonCallInstructionData
	{
		RAIIStoreNonCallInstructionData();
		~RAIIStoreNonCallInstructionData();

		py::Value reg0;
	};

  public:
	virtual ~Instruction() = default;
	virtual std::string to_string() const = 0;
	virtual py::PyResult<py::Value> execute(VirtualMachine &, Interpreter &) const = 0;
	virtual void relocate(std::size_t) = 0;
	virtual std::vector<std::uint8_t> serialize() const = 0;
	virtual std::uint8_t id() const = 0;
};

export constexpr std::uint8_t BINARY_OPERATION = 0;
export constexpr std::uint8_t BINARY_SUBSCRIPT = 1;
export constexpr std::uint8_t BUILD_DICT = 2;
export constexpr std::uint8_t BUILD_LIST = 3;
export constexpr std::uint8_t BUILD_TUPLE = 4;
export constexpr std::uint8_t CLEAR_EXCEPTION_STATE = 5;
export constexpr std::uint8_t COMPARE_OP = 6;
export constexpr std::uint8_t DELETE_NAME = 7;
export constexpr std::uint8_t DICT_MERGE = 8;
export constexpr std::uint8_t FOR_ITER = 9;
export constexpr std::uint8_t FUNCTION_CALL = 10;
export constexpr std::uint8_t FUNCTION_CALL_EX = 11;
export constexpr std::uint8_t FUNCTION_CALL_WITH_KW = 12;
export constexpr std::uint8_t GET_ITER = 13;
export constexpr std::uint8_t IMPORT_NAME = 14;
export constexpr std::uint8_t JUMP = 17;
export constexpr std::uint8_t JUMP_FORWARD = 18;
export constexpr std::uint8_t JUMP_IF_FALSE = 19;
export constexpr std::uint8_t JUMP_IF_FALSE_OR_POP = 20;
export constexpr std::uint8_t JUMP_IF_NOT_EXCEPTION_MATCH = 21;
export constexpr std::uint8_t JUMP_IF_TRUE = 22;
export constexpr std::uint8_t JUMP_IF_TRUE_OR_POP = 23;
export constexpr std::uint8_t LIST_EXTEND = 24;
export constexpr std::uint8_t LIST_TO_TUPLE = 25;
export constexpr std::uint8_t LOAD_ASSERTION_ERROR = 26;
export constexpr std::uint8_t LOAD_ATTR = 27;
export constexpr std::uint8_t LOAD_BUILD_CLASS = 28;
export constexpr std::uint8_t LOAD_CLOSURE = 29;
export constexpr std::uint8_t LOAD_CONST = 30;
export constexpr std::uint8_t LOAD_DEREF = 31;
export constexpr std::uint8_t LOAD_FAST = 32;
export constexpr std::uint8_t LOAD_GLOBAL = 33;
export constexpr std::uint8_t LOAD_METHOD = 34;
export constexpr std::uint8_t LOAD_NAME = 35;
export constexpr std::uint8_t MAKE_FUNCTION = 36;
export constexpr std::uint8_t METHOD_CALL = 38;
export constexpr std::uint8_t MOVE = 39;
export constexpr std::uint8_t RAISE_VARARGS = 40;
export constexpr std::uint8_t RETURN_VALUE = 41;
export constexpr std::uint8_t SETUP_EXCEPTION_HANDLING = 42;
export constexpr std::uint8_t STORE_ATTR = 43;
export constexpr std::uint8_t STORE_DEREF = 44;
export constexpr std::uint8_t STORE_FAST = 45;
export constexpr std::uint8_t STORE_GLOBAL = 46;
export constexpr std::uint8_t STORE_NAME = 47;
export constexpr std::uint8_t STORE_SUBSCRIPT = 48;
export constexpr std::uint8_t UNARY = 49;
export constexpr std::uint8_t UNPACK_SEQUENCE = 50;
export constexpr std::uint8_t CONTINUE = 51;
export constexpr std::uint8_t RERAISE = 52;
export constexpr std::uint8_t WITH_EXCEPT_START = 53;
export constexpr std::uint8_t LEAVE_EXCEPTION_HANDLING = 54;
export constexpr std::uint8_t DELETE_SUBSCRIPT = 55;
export constexpr std::uint8_t SETUP_WITH = 56;
export constexpr std::uint8_t CLEAR_TOP_CLEANUP = 57;
export constexpr std::uint8_t LIST_APPEND = 58;
export constexpr std::uint8_t SET_ADD = 59;
export constexpr std::uint8_t BUILD_SET = 60;
export constexpr std::uint8_t IMPORT_FROM = 61;
export constexpr std::uint8_t BUILD_SLICE = 62;
export constexpr std::uint8_t INPLACE_OP = 63;
export constexpr std::uint8_t YIELD_VALUE = 64;
export constexpr std::uint8_t DICT_UPDATE = 65;
export constexpr std::uint8_t DICT_ADD = 66;
export constexpr std::uint8_t YIELD_LOAD = 67;
export constexpr std::uint8_t GET_YIELD_FROM_ITER = 68;
export constexpr std::uint8_t YIELD_FROM = 69;
export constexpr std::uint8_t IMPORT_STAR = 70;
export constexpr std::uint8_t GET_AWAITABLE = 71;
export constexpr std::uint8_t BUILD_STRING = 72;
export constexpr std::uint8_t FORMAT_VALUE = 73;
export constexpr std::uint8_t PUSH = 74;
export constexpr std::uint8_t POP = 75;
export constexpr std::uint8_t DELETE_FAST = 76;
export constexpr std::uint8_t DELETE_GLOBAL = 77;
export constexpr std::uint8_t JUMP_IF_EXCEPTION_MATCH = 78;
export constexpr std::uint8_t TO_BOOL = 79;
export constexpr std::uint8_t SET_UPDATE = 80;
export constexpr std::uint8_t DELETE_ATTR = 81;
export constexpr std::uint8_t UNPACK_EXPAND = 82;
export constexpr std::uint8_t DELETE_DEREF = 83;
export constexpr std::uint8_t LOAD_EXCEPTION = 84;

export std::unique_ptr<Instruction> deserialize(std::span<const std::uint8_t> &instruction_buffer);

// The concrete instructions live in this partition too. As plain headers they
// were included from py.codegen's purview (BytecodeGenerator) and py.runtime's
// (BuiltinsModule) and from ordinary TUs, so each class was attached to two
// modules at once and its execute()/serialize() were undefined in both.
export {
#include "BinaryOperation.hpp"
#include "BinarySubscript.hpp"
#include "BuildDict.hpp"
#include "BuildList.hpp"
#include "BuildSet.hpp"
#include "BuildSlice.hpp"
#include "BuildString.hpp"
#include "BuildTuple.hpp"
#include "ClearExceptionState.hpp"
#include "ClearTopCleanup.hpp"
#include "CompareOperation.hpp"
#include "DeleteAttr.hpp"
#include "DeleteDeref.hpp"
#include "DeleteFast.hpp"
#include "DeleteGlobal.hpp"
#include "DeleteName.hpp"
#include "DeleteSubscript.hpp"
#include "DictAdd.hpp"
#include "DictMerge.hpp"
#include "DictUpdate.hpp"
#include "ForIter.hpp"
#include "FormatValue.hpp"
#include "FunctionCall.hpp"
#include "FunctionCallEx.hpp"
#include "FunctionCallWithKeywords.hpp"
#include "GetAwaitable.hpp"
#include "GetIter.hpp"
#include "GetYieldFromIter.hpp"
#include "ImportFrom.hpp"
#include "ImportName.hpp"
#include "ImportStar.hpp"
#include "InplaceOp.hpp"
#include "Jump.hpp"
#include "JumpForward.hpp"
#include "JumpIfExceptionMatch.hpp"
#include "JumpIfFalse.hpp"
#include "JumpIfFalseOrPop.hpp"
#include "JumpIfNotExceptionMatch.hpp"
#include "JumpIfTrue.hpp"
#include "JumpIfTrueOrPop.hpp"
#include "LeaveExceptionHandling.hpp"
#include "ListAppend.hpp"
#include "ListExtend.hpp"
#include "ListToTuple.hpp"
#include "LoadAssertionError.hpp"
#include "LoadAttr.hpp"
#include "LoadBuildClass.hpp"
#include "LoadClosure.hpp"
#include "LoadConst.hpp"
#include "LoadDeref.hpp"
#include "LoadException.hpp"
#include "LoadFast.hpp"
#include "LoadGlobal.hpp"
#include "LoadMethod.hpp"
#include "LoadName.hpp"
#include "MakeFunction.hpp"
#include "MethodCall.hpp"
#include "Move.hpp"
#include "Pop.hpp"
#include "Push.hpp"
#include "RaiseVarargs.hpp"
#include "ReRaise.hpp"
#include "ReturnValue.hpp"
#include "SetAdd.hpp"
#include "SetUpdate.hpp"
#include "SetupExceptionHandling.hpp"
#include "SetupWith.hpp"
#include "StoreAttr.hpp"
#include "StoreDeref.hpp"
#include "StoreFast.hpp"
#include "StoreGlobal.hpp"
#include "StoreName.hpp"
#include "StoreSubscript.hpp"
#include "ToBool.hpp"
#include "Unary.hpp"
#include "UnpackExpand.hpp"
#include "UnpackSequence.hpp"
#include "WithExceptStart.hpp"
#include "YieldFrom.hpp"
#include "YieldLoad.hpp"
#include "YieldValue.hpp"
}
