#include "InplaceOp.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> InplaceOp::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);
	auto result = [this, &lhs, &rhs, &interpreter] {
		switch (m_operation) {
		case Operation::PLUS: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return add(lhs, rhs, interpreter);
		} break;
		case Operation::MINUS: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return subtract(lhs, rhs, interpreter);
		} break;
		case Operation::MODULO: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return modulo(lhs, rhs, interpreter);
		} break;
		case Operation::MULTIPLY: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return multiply(lhs, rhs, interpreter);
		} break;
		case Operation::EXP: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return exp(lhs, rhs, interpreter);
		} break;
		case Operation::SLASH: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return true_divide(lhs, rhs, interpreter);
		} break;
		case Operation::FLOORDIV: {
			TODO();
		} break;
		case Operation::MATMUL: {
			TODO();
		} break;
		case Operation::LEFTSHIFT: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return lshift(lhs, rhs, interpreter);
		} break;
		case Operation::RIGHTSHIFT: {
			TODO();
		} break;
		case Operation::AND: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return and_(lhs, rhs, interpreter);
		} break;
		case Operation::OR: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return or_(lhs, rhs, interpreter);
		} break;
		case Operation::XOR: {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return xor_(lhs, rhs, interpreter);
		} break;
		}
		ASSERT_NOT_REACHED();
	}();
	if (result.is_ok()) { lhs = result.unwrap(); }
	return result;
}

std::vector<uint8_t> InplaceOp::serialize() const
{
	return {
		INPLACE_OP,
		m_lhs,
		m_rhs,
		static_cast<uint8_t>(m_operation),
	};
}
