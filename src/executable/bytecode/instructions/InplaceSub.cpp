#include "InplaceSub.hpp"
#include "runtime/Value.hpp"

void InplaceSub::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);
	if (auto result = subtract(lhs, rhs, interpreter)) { lhs = *result; }
}