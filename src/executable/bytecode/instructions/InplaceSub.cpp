#include "InplaceSub.hpp"
#include "runtime/Value.hpp"

py::PyResult InplaceSub::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);
	auto result = subtract(lhs, rhs, interpreter);
	if (result.is_ok()) { lhs = result.unwrap(); }
	return result;
}

std::vector<uint8_t> InplaceSub::serialize() const
{
	return {
		INPLACE_SUB,
		m_lhs,
		m_rhs,
	};
}