#include "ClearExceptionState.hpp"
#include "runtime/PyNone.hpp"

using namespace py;


PyResult ClearExceptionState::execute(VirtualMachine &, Interpreter &interpreter) const
{
	interpreter.execution_frame()->pop_exception();

	return PyResult::Ok(py_none());
}

std::vector<uint8_t> ClearExceptionState::serialize() const { return { CLEAR_EXCEPTION_STATE }; }
