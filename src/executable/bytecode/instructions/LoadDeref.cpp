#include "LoadDeref.hpp"
#include "runtime/PyCell.hpp"

using namespace py;

void LoadDeref::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->freevars().size() > m_source)
	vm.reg(m_destination) = interpreter.execution_frame()->freevars()[m_source]->content();
}