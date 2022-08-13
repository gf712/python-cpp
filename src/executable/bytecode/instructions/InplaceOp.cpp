#include "InplaceOp.hpp"
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
			return add(lhs, rhs, interpreter);
		} break;
		case Operation::MINUS: {
			return subtract(lhs, rhs, interpreter);
		} break;
		case Operation::MODULO: {
			return modulo(lhs, rhs, interpreter);
		} break;
		case Operation::MULTIPLY: {
			return multiply(lhs, rhs, interpreter);
		} break;
		case Operation::EXP: {
			return exp(lhs, rhs, interpreter);
		} break;
		case Operation::SLASH: {
			return true_divide(lhs, rhs, interpreter);
		} break;
		case Operation::FLOORDIV: {
			TODO();
		} break;
		case Operation::LEFTSHIFT: {
			return lshift(lhs, rhs, interpreter);
		} break;
		case Operation::RIGHTSHIFT: {
			TODO();
		} break;
		case Operation::AND: {
			TODO();
		} break;
		case Operation::OR: {
			TODO();
		} break;
		case Operation::XOR: {
			TODO();
		} break;
		}
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