#include "StoreDeref.hpp"
#include "runtime/PyCell.hpp"

using namespace py;

void StoreDeref::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->freevars().size() > m_dst)
	interpreter.execution_frame()->freevars()[m_dst] = PyCell::create(vm.reg(m_src));
}
