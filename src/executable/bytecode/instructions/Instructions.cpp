#include "Instructions.hpp"
#include "FunctionCall.hpp"
#include "LoadConst.hpp"
#include "LoadName.hpp"
#include "Move.hpp"
#include "ReturnValue.hpp"

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
	}
	TODO();
}