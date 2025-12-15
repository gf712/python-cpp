#include "ReRaise.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"

using namespace py;


PyResult<Value> ReRaise::execute(VirtualMachine &, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->exception_info().has_value());
	return Err(interpreter.execution_frame()->pop_exception());
}

std::vector<uint8_t> ReRaise::serialize() const { return { RERAISE }; }
