#include "SetupExceptionHandling.hpp"
#include "runtime/PyNone.hpp"

using namespace py;

PyResult<Value> SetupExceptionHandling::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.set_exception_handling();
	return Ok(Value{ py_none() });
}

std::vector<uint8_t> SetupExceptionHandling::serialize() const
{
	return {
		SETUP_EXCEPTION_HANDLING,
	};
}
