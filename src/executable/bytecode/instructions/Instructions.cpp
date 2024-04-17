#include "Instructions.hpp"
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
#include "DeleteName.hpp"
#include "DeleteSubscript.hpp"
#include "DictMerge.hpp"
#include "ForIter.hpp"
#include "FormatValue.hpp"
#include "FunctionCall.hpp"
#include "FunctionCallEx.hpp"
#include "FunctionCallWithKeywords.hpp"
#include "GetIter.hpp"
#include "ImportFrom.hpp"
#include "ImportName.hpp"
#include "InplaceOp.hpp"
#include "Jump.hpp"
#include "JumpForward.hpp"
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
#include "SetupExceptionHandling.hpp"
#include "SetupWith.hpp"
#include "StoreAttr.hpp"
#include "StoreDeref.hpp"
#include "StoreFast.hpp"
#include "StoreGlobal.hpp"
#include "StoreName.hpp"
#include "StoreSubscript.hpp"
#include "Unary.hpp"
#include "UnpackSequence.hpp"
#include "WithExceptStart.hpp"
#include "YieldValue.hpp"

#include "../serialization/deserialize.hpp"

#include <optional>

using namespace py;

std::unique_ptr<Instruction> deserialize(std::span<const uint8_t> &instruction_buffer)
{
	const auto instruction_code = deserialize<uint8_t>(instruction_buffer);
	switch (instruction_code) {
	case LOAD_NAME: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto object_name = deserialize<std::string>(instruction_buffer);
		return std::make_unique<LoadName>(dst, object_name);
	} break;
	case LOAD_CONST: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto static_value_index = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<LoadConst>(dst, static_value_index);
	} break;
	case FUNCTION_CALL: {
		const auto function_name_reg = deserialize<uint8_t>(instruction_buffer);
		const auto args_count = deserialize<uint8_t>(instruction_buffer);
		const auto args_offset = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<FunctionCall>(function_name_reg, args_count, args_offset);
	} break;
	case RETURN_VALUE: {
		const auto source = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<ReturnValue>(source);
	} break;
	case MOVE: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<Move>(dst, src);
	} break;
	case MAKE_FUNCTION: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto name = deserialize<uint8_t>(instruction_buffer);
		const auto default_size = deserialize<uint8_t>(instruction_buffer);
		const auto kw_default_size = deserialize<uint8_t>(instruction_buffer);
		const auto has_captures_tuple = deserialize<uint8_t>(instruction_buffer);
		const auto captures_tuple = [&]() -> std::optional<Register> {
			const auto value = deserialize<uint8_t>(instruction_buffer);
			if (has_captures_tuple) {
				return value;
			} else {
				return std::nullopt;
			}
		}();
		return std::make_unique<MakeFunction>(
			dst, name, default_size, kw_default_size, captures_tuple);
	} break;
	case STORE_NAME: {
		const auto src = deserialize<uint8_t>(instruction_buffer);
		const auto name = deserialize<std::string>(instruction_buffer);
		return std::make_unique<StoreName>(name, src);
	} break;
	case COMPARE_OP: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto lhs = deserialize<uint8_t>(instruction_buffer);
		const auto rhs = deserialize<uint8_t>(instruction_buffer);
		const auto comparisson =
			static_cast<CompareOperation::Comparisson>(deserialize<uint8_t>(instruction_buffer));
		return std::make_unique<CompareOperation>(dst, lhs, rhs, comparisson);
	} break;
	case JUMP_IF_TRUE: {
		const auto test_register = deserialize<uint8_t>(instruction_buffer);
		const auto offset = deserialize<int32_t>(instruction_buffer);
		return std::make_unique<JumpIfTrue>(test_register, offset);
	} break;
	case RAISE_VARARGS: {
		uint8_t count = deserialize<uint8_t>(instruction_buffer);
		const auto exception = [&]() -> std::optional<Register> {
			if (count > 0) { return deserialize<uint8_t>(instruction_buffer); }
			return std::nullopt;
		}();
		const auto cause = [&]() -> std::optional<Register> {
			if (count > 1) { return deserialize<uint8_t>(instruction_buffer); }
			return std::nullopt;
		}();
		if (exception.has_value() && cause.has_value()) {
			return std::make_unique<RaiseVarargs>(*exception, *cause);
		} else if (exception.has_value()) {
			return std::make_unique<RaiseVarargs>(*exception);
		} else {
			return std::make_unique<RaiseVarargs>();
		}
	} break;
	case LOAD_ASSERTION_ERROR: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<LoadAssertionError>(dst);
	} break;
	case LOAD_FAST: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto idx = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<LoadFast>(dst, idx, "NOT_IMPLEMENTED");
	} break;
	case BINARY_OPERATION: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto lhs = deserialize<uint8_t>(instruction_buffer);
		const auto rhs = deserialize<uint8_t>(instruction_buffer);
		const auto operation =
			static_cast<BinaryOperation::Operation>(deserialize<uint8_t>(instruction_buffer));
		return std::make_unique<BinaryOperation>(dst, lhs, rhs, operation);
	} break;
	case LOAD_ATTR: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto value_source = deserialize<uint8_t>(instruction_buffer);
		const auto attr_name = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<LoadAttr>(dst, value_source, attr_name);
	}
	case LOAD_GLOBAL: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto obj_name = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<LoadGlobal>(dst, obj_name);
	}
	case RERAISE: {
		return std::make_unique<ReRaise>();
	}
	case LOAD_METHOD: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto value_source = deserialize<uint8_t>(instruction_buffer);
		const auto method_name = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<LoadMethod>(dst, value_source, method_name);
	}
	case METHOD_CALL: {
		const auto caller = deserialize<uint8_t>(instruction_buffer);
		auto args = deserialize<std::vector<uint8_t>>(instruction_buffer);
		return std::make_unique<MethodCall>(caller, std::move(args));
	}
	case STORE_ATTR: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		const auto attr_name = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<StoreAttr>(dst, src, attr_name);
	}
	case STORE_FAST: {
		const auto stack_index = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<StoreFast>(stack_index, src);
	}
	case WITH_EXCEPT_START: {
		const auto result = deserialize<uint8_t>(instruction_buffer);
		const auto exit_method = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<WithExceptStart>(result, exit_method);
	}
	case LOAD_CLOSURE: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<LoadClosure>(dst, src);
	}
	case FUNCTION_CALL_WITH_KW: {
		const auto func_name = deserialize<uint8_t>(instruction_buffer);
		auto args = deserialize<std::vector<uint8_t>>(instruction_buffer);
		auto kwargs = deserialize<std::vector<uint8_t>>(instruction_buffer);
		auto keywords = deserialize<std::vector<uint8_t>>(instruction_buffer);
		return std::make_unique<FunctionCallWithKeywords>(
			func_name, std::move(args), std::move(kwargs), std::move(keywords));
	}
	case STORE_GLOBAL: {
		const auto name = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<StoreGlobal>(name, src);
	}
	case IMPORT_NAME: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto name = deserialize<uint8_t>(instruction_buffer);
		const auto from_list = deserialize<uint8_t>(instruction_buffer);
		const auto level = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<ImportName>(dst, name, from_list, level);
	}
	case IMPORT_FROM: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto name = deserialize<uint8_t>(instruction_buffer);
		const auto from = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<ImportFrom>(dst, name, from);
	}
	case BUILD_DICT: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto size = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<BuildDict>(dst, size);
	}
	case BUILD_SLICE: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto argcount = deserialize<uint8_t>(instruction_buffer);
		ASSERT(argcount <= 3);
		ASSERT(argcount > 1);
		if (argcount == 2) {
			const auto start = deserialize<uint8_t>(instruction_buffer);
			const auto end = deserialize<uint8_t>(instruction_buffer);
			return std::make_unique<BuildSlice>(dst, start, end);
		} else {
			const auto start = deserialize<uint8_t>(instruction_buffer);
			const auto end = deserialize<uint8_t>(instruction_buffer);
			const auto step = deserialize<uint8_t>(instruction_buffer);
			return std::make_unique<BuildSlice>(dst, start, end, step);
		}
	}
	case CLEAR_EXCEPTION_STATE: {
		return std::make_unique<ClearExceptionState>();
	}
	case DELETE_NAME: {
		const auto name = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<DeleteName>(name);
	}
	case LIST_EXTEND: {
		const auto list = deserialize<uint8_t>(instruction_buffer);
		const auto value = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<ListExtend>(list, value);
	}
	case FOR_ITER: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		const auto offset = deserialize<int32_t>(instruction_buffer);
		const auto body_offset = deserialize<int32_t>(instruction_buffer);
		return std::make_unique<ForIter>(dst, src, offset, body_offset);
	}
	case GET_ITER: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<GetIter>(dst, src);
	}
	case LOAD_BUILD_CLASS: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<LoadBuildClass>(dst);
	}
	case BUILD_TUPLE: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto size = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<BuildTuple>(dst, size);
	}
	case BUILD_LIST: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto size = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<BuildList>(dst, size);
	}
	case SETUP_EXCEPTION_HANDLING: {
		const auto offset = deserialize<uint32_t>(instruction_buffer);
		return std::make_unique<SetupExceptionHandling>(offset);
	}
	case JUMP_FORWARD: {
		const auto offset = deserialize<uint32_t>(instruction_buffer);
		return std::make_unique<JumpForward>(offset);
	}
	case JUMP_IF_NOT_EXCEPTION_MATCH: {
		const auto exception = deserialize<uint8_t>(instruction_buffer);
		const auto offset = deserialize<uint32_t>(instruction_buffer);
		return std::make_unique<JumpIfNotExceptionMatch>(exception, offset);
	}
	case JUMP_IF_FALSE: {
		const auto test_register = deserialize<uint8_t>(instruction_buffer);
		const auto offset = deserialize<int32_t>(instruction_buffer);
		return std::make_unique<JumpIfFalse>(test_register, offset);
	}
	case JUMP: {
		const auto offset = deserialize<int32_t>(instruction_buffer);
		return std::make_unique<Jump>(offset);
	}
	case STORE_SUBSCRIPT: {
		const auto obj = deserialize<uint8_t>(instruction_buffer);
		const auto slice = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<StoreSubscript>(obj, slice, src);
	}
	case BINARY_SUBSCRIPT: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		const auto index = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<BinarySubscript>(dst, src, index);
	}
	case JUMP_IF_TRUE_OR_POP: {
		const auto test_register = deserialize<uint8_t>(instruction_buffer);
		const auto result_register = deserialize<uint8_t>(instruction_buffer);
		const auto offset = deserialize<int32_t>(instruction_buffer);
		return std::make_unique<JumpIfTrueOrPop>(test_register, result_register, offset);
	}
	case JUMP_IF_FALSE_OR_POP: {
		const auto test_register = deserialize<uint8_t>(instruction_buffer);
		const auto result_register = deserialize<uint8_t>(instruction_buffer);
		const auto offset = deserialize<int32_t>(instruction_buffer);
		return std::make_unique<JumpIfFalseOrPop>(test_register, result_register, offset);
	}
	case INPLACE_OP: {
		const auto lhs = deserialize<uint8_t>(instruction_buffer);
		const auto rhs = deserialize<uint8_t>(instruction_buffer);
		const auto op = static_cast<InplaceOp::Operation>(deserialize<uint8_t>(instruction_buffer));
		return std::make_unique<InplaceOp>(lhs, rhs, op);
	}
	case LIST_TO_TUPLE: {
		const auto tuple = deserialize<uint8_t>(instruction_buffer);
		const auto list = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<ListToTuple>(tuple, list);
	}
	case DICT_MERGE: {
		const auto this_dict = deserialize<uint8_t>(instruction_buffer);
		const auto other_dict = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<DictMerge>(this_dict, other_dict);
	}
	case FUNCTION_CALL_EX: {
		const auto function = deserialize<uint8_t>(instruction_buffer);
		const auto args = deserialize<uint8_t>(instruction_buffer);
		const auto kwargs = deserialize<uint8_t>(instruction_buffer);
		const auto expand_args = deserialize<bool>(instruction_buffer);
		const auto expand_kwargs = deserialize<bool>(instruction_buffer);
		return std::make_unique<FunctionCallEx>(function, args, kwargs, expand_args, expand_kwargs);
	}
	case UNARY: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		const auto op = deserialize<uint8_t>(instruction_buffer);
		ASSERT(op < 4)
		return std::make_unique<Unary>(dst, src, static_cast<Unary::Operation>(op));
	}
	case LOAD_DEREF: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<LoadDeref>(dst, src);
	}
	case LEAVE_EXCEPTION_HANDLING: {
		return std::make_unique<LeaveExceptionHandling>();
	}
	case DELETE_SUBSCRIPT: {
		const auto value = deserialize<uint8_t>(instruction_buffer);
		const auto index = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<DeleteSubscript>(value, index);
	}
	case SETUP_WITH: {
		const auto offset = deserialize<uint32_t>(instruction_buffer);
		return std::make_unique<SetupWith>(offset);
	}
	case CLEAR_TOP_CLEANUP: {
		return std::make_unique<ClearTopCleanup>();
	}
	case UNPACK_SEQUENCE: {
		const auto dst = deserialize<std::vector<uint8_t>>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<UnpackSequence>(dst, src);
	}
	case LIST_APPEND: {
		const auto list = deserialize<uint8_t>(instruction_buffer);
		const auto value = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<ListAppend>(list, value);
	}
	case SET_ADD: {
		const auto set = deserialize<uint8_t>(instruction_buffer);
		const auto value = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<SetAdd>(set, value);
	}
	case BUILD_SET: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto size = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<BuildSet>(dst, size);
	}
	case STORE_DEREF: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<StoreDeref>(dst, src);
	}
	case YIELD_VALUE: {
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<YieldValue>(src);
	}
	case BUILD_STRING: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto size = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<BuildString>(dst, size);
	}
	case FORMAT_VALUE: {
		const auto dst = deserialize<uint8_t>(instruction_buffer);
		const auto src = deserialize<uint8_t>(instruction_buffer);
		const auto conversion = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<FormatValue>(dst, src, conversion);
	}
	case PUSH: {
		const auto src = deserialize<uint8_t>(instruction_buffer);
		return std::make_unique<Push>(src);
	}
	case POP: {
		return std::make_unique<Pop>();
	}
	}
	spdlog::error("Missing opcode: {}", instruction_code);
	TODO();
}