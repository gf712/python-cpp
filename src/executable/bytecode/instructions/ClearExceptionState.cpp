#include "ClearExceptionState.hpp"
#include "runtime/PyNone.hpp"

using namespace py;

namespace {
bool has_stashed_exception(Interpreter &interpreter)
{
	return interpreter.execution_frame()->stashed_exception_info().has_value();
}
}// namespace

PyResult ClearExceptionState::execute(VirtualMachine &, Interpreter &interpreter) const
{
	ASSERT(has_stashed_exception(interpreter))

	interpreter.execution_frame()->clear_stashed_exception();

	return PyResult::Ok(py_none());
}

std::vector<uint8_t> ClearExceptionState::serialize() const { return { CLEAR_EXCEPTION_STATE }; }
