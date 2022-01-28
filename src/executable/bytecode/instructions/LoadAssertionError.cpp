#include "LoadAssertionError.hpp"
#include "runtime/AssertionError.hpp"
#include "runtime/PyType.hpp"

void LoadAssertionError::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.reg(m_assertion_location) = AssertionError::this_type();
}