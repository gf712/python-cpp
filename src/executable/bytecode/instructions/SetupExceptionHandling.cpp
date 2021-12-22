#include "SetupExceptionHandling.hpp"


void SetupExceptionHandling::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.set_exception_handling();
}
