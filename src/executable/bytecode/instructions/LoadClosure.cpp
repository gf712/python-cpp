#include "LoadClosure.hpp"
#include "runtime/PyCell.hpp"


using namespace py;

void LoadClosure::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	// size_t freevars_stack_offset =
	// 	vm.stack_locals()->get().size() - interpreter.execution_frame()->freevars().size();
	// const auto &value = vm.stack_local(freevars_stack_offset + m_source);
	vm.reg(m_destination) = interpreter.execution_frame()->freevars()[m_source];
}