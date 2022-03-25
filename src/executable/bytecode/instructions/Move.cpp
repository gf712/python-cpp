#include "Move.hpp"

void Move::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.reg(m_destination) = vm.reg(m_source);
}