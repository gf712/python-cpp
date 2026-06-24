#include "ReRaise.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/RuntimeError.hpp"

using namespace py;


PyResult<Value> ReRaise::execute(VirtualMachine &, Interpreter &interpreter) const
{
	// A bare `raise` with no exception currently being handled is a RuntimeError,
	// not an abort. (Now that the exception stack is kept clean, the stack is
	// genuinely empty here rather than holding a stale internal exception.)
	if (!interpreter.execution_frame()->exception_info().has_value()) {
		return Err(runtime_error("No active exception to reraise"));
	}
	return Err(interpreter.execution_frame()->pop_exception());
}

std::vector<uint8_t> ReRaise::serialize() const { return { RERAISE }; }
