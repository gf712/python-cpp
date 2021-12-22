#include "ClearExceptionState.hpp"

namespace {
bool has_active_exception(Interpreter &interpreter)
{
	return interpreter.execution_frame()->exception()
		   || interpreter.status() == Interpreter::Status::EXCEPTION;
}
}// namespace

void ClearExceptionState::execute(VirtualMachine &, Interpreter &interpreter) const
{
	ASSERT(has_active_exception(interpreter))

	if (!interpreter.execution_frame()->exception()) {
		// TODO: this is a deprecated API, need to remove usage
		TODO();
	}

	interpreter.execution_frame()->set_exception(nullptr);
}