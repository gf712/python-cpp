#include "CompareOperation.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/Value.hpp"

using namespace py;

PyResult CompareOperation::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);

	const auto result = [&]() -> PyResult {
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
			return is(lhs, rhs, interpreter);
		} break;
		case Comparisson::IsNot: {
			if (auto result = is(lhs, rhs, interpreter); result.is_ok()) {
				if (std::holds_alternative<NameConstant>(result.unwrap())) {
					ASSERT(
						std::holds_alternative<bool>(std::get<NameConstant>(result.unwrap()).value))
					return PyResult::Ok(NameConstant{
						!std::get<bool>(std::get<NameConstant>(result.unwrap()).value) });
				} else if (std::holds_alternative<PyObject *>(result.unwrap())) {
					return PyResult::Ok(
						result.unwrap_as<PyObject>() == py_false() ? py_true() : py_false());
				} else {
					TODO();
				}
			} else {
				return result;
			}
		} break;
		case Comparisson::In: {
			return in(lhs, rhs, interpreter);
		} break;
		case Comparisson::NotIn: {
			if (auto result = in(lhs, rhs, interpreter); result.is_ok()) {
				if (std::holds_alternative<NameConstant>(result.unwrap())) {
					ASSERT(
						std::holds_alternative<bool>(std::get<NameConstant>(result.unwrap()).value))
					return PyResult::Ok(NameConstant{
						!std::get<bool>(std::get<NameConstant>(result.unwrap()).value) });
				} else if (std::holds_alternative<PyObject *>(result.unwrap())) {
					return PyResult::Ok(
						result.unwrap_as<PyObject>() == py_false() ? py_true() : py_false());
				} else {
					TODO();
				}
			} else {
				return result;
			}
		} break;
		}
	}();

	if (result.is_ok()) {
		ASSERT(vm.registers().has_value())
		ASSERT(vm.registers()->get().size() > m_dst)
		vm.reg(m_dst) = result.unwrap();
	}
	return result;
};

std::vector<uint8_t> CompareOperation::serialize() const
{
	return {
		COMPARE_OP,
		m_dst,
		m_lhs,
		m_rhs,
		static_cast<uint8_t>(m_comparisson),
	};
}
