module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


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
