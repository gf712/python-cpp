#include "Instructions.hpp"
#include "BinaryOperation.hpp"
#include "CompareOperation.hpp"
#include "FunctionCall.hpp"
#include "JumpIfTrue.hpp"
#include "LoadAssertionError.hpp"
#include "LoadConst.hpp"
#include "LoadFast.hpp"
#include "LoadName.hpp"
#include "MakeFunction.hpp"
#include "Move.hpp"
#include "RaiseVarargs.hpp"
#include "ReturnValue.hpp"
#include "StoreName.hpp"

#include "../serialization/deserialize.hpp"

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
		const auto default_stack_offset = deserialize<uint8_t>(instruction_buffer);
		const auto kw_default_size = deserialize<uint8_t>(instruction_buffer);
		const auto kw_default_stack_offset = deserialize<uint8_t>(instruction_buffer);
		const auto has_captures_tuple = deserialize<uint8_t>(instruction_buffer);
		const auto captures_tuple = [&]() -> std::optional<Register> {
			const auto value = deserialize<uint8_t>(instruction_buffer);
			if (has_captures_tuple) {
				return value;
			} else {
				return std::nullopt;
			}
		}();
		return std::make_unique<MakeFunction>(dst,
			name,
			default_size,
			default_stack_offset,
			kw_default_size,
			kw_default_stack_offset,
			captures_tuple);
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
		const auto label_position = deserialize<uint8_t>(instruction_buffer);
		const auto offset = deserialize<uint8_t>(instruction_buffer);
		(void)offset;
		return std::make_unique<JumpIfTrue>(test_register, std::make_shared<Label>(label_position));
	} break;
	case RAISE_VARARGS: {
		const auto exception = [&]() -> std::optional<Register> {
			auto value = deserialize<uint8_t>(instruction_buffer);
			if (value == 0) {
				return std::nullopt;
			} else {
				return value;
			}
		}();
		const auto cause = [&]() -> std::optional<Register> {
			auto value = deserialize<uint8_t>(instruction_buffer);
			if (value == 0) {
				return std::nullopt;
			} else {
				return value;
			}
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
	}
	TODO();
}