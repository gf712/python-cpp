#include "CompareOperation.hpp"
#include "runtime/Value.hpp"

using namespace py;

void CompareOperation::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);

	const auto result = [&]() -> std::optional<Value> {
		switch (m_comparisson) {
		case Comparisson::Eq: {
			return equals(lhs, rhs, interpreter);
		} break;
		case Comparisson::NotEq: {
			return not_equals(lhs, rhs, interpreter);
		} break;
		case Comparisson::Lt: {
			return less_than(lhs, rhs, interpreter);
		} break;
		case Comparisson::LtE: {
			return less_than_equals(lhs, rhs, interpreter);
		} break;
		case Comparisson::Gt: {
			return greater_than(lhs, rhs, interpreter);
		} break;
		case Comparisson::GtE: {
			return greater_than_equals(lhs, rhs, interpreter);
		} break;
		case Comparisson::Is: {
			return NameConstant{ is(lhs, rhs, interpreter) };
		} break;
		case Comparisson::IsNot: {
			return NameConstant{ !is(lhs, rhs, interpreter) };
		} break;
		case Comparisson::In: {
			return NameConstant{ in(lhs, rhs, interpreter) };
		} break;
		case Comparisson::NotIn: {
			return NameConstant{ !in(lhs, rhs, interpreter) };
		} break;
		}
	}();

	if (result) {
		ASSERT(vm.registers().has_value())
		ASSERT(vm.registers()->get().size() > m_dst)
		vm.reg(m_dst) = *result;
	}
};