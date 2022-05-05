#include "ClearExceptionState.hpp"
#include "runtime/PyNone.hpp"

using namespace py;


PyResult<Value> ClearExceptionState::execute(VirtualMachine &, Interpreter &interpreter) const
{
	interpreter.execution_frame()->pop_exception();

	return Ok(Value{ py_none() });
}

std::vector<uint8_t> ClearExceptionState::serialize() const { return { CLEAR_EXCEPTION_STATE }; }
