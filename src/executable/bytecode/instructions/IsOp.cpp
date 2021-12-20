#include "IsOp.hpp"
#include "runtime/Value.hpp"

void IsOp::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);
	const bool result = is(lhs, rhs, interpreter) ^ m_is_not;

	ASSERT(vm.registers().has_value())
	ASSERT(vm.registers()->get().size() > m_dst)
	vm.reg(m_dst) = NameConstant{ result };
}