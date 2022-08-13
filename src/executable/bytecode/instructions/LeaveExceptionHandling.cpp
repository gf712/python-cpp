#include "LeaveExceptionHandling.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LeaveExceptionHandling::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.leave_cleanup_handling();
	return Ok(Value{ py_none() });
}

std::vector<uint8_t> LeaveExceptionHandling::serialize() const
{
	return {
		LEAVE_EXCEPTION_HANDLING,
	};
}
