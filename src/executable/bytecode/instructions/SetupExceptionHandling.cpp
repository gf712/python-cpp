#include "SetupExceptionHandling.hpp"


void SetupExceptionHandling::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.set_exception_handling();
}

std::vector<uint8_t> SetupExceptionHandling::serialize() const
{
	return {
		SETUP_EXCECPTION_HANDLING,
	};
}
