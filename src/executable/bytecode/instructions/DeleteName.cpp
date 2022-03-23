#include "DeleteName.hpp"

void DeleteName::execute(VirtualMachine &vm, Interpreter &) const
{
	auto obj = vm.reg(m_name);
    TODO();
}