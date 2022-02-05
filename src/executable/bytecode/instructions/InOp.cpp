#include "InOp.hpp"
#include "runtime/Value.hpp"

using namespace py;

void InOp::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);
	const bool result = in(lhs, rhs, interpreter) ^ m_not_in;

	ASSERT(vm.registers().has_value())
	ASSERT(vm.registers()->get().size() > m_dst)
	vm.reg(m_dst) = NameConstant{ result };
}