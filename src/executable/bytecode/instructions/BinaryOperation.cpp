#include "BinaryOperation.hpp"
#include "runtime/Value.hpp"

using namespace py;

PyResult BinaryOperation::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);

	const auto result = [&]() {
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
		}
	}();

	if (result.is_ok()) {
		ASSERT(vm.registers().has_value())
		ASSERT(m_destination < vm.registers()->get().size())
		vm.reg(m_destination) = result.unwrap();
	}
	return result;
}

std::vector<uint8_t> BinaryOperation::serialize() const
{
	return {
		BINARY_OPERATION,
		m_destination,
		m_lhs,
		m_rhs,
		static_cast<uint8_t>(m_operation),
	};
}
