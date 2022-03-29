#include "ClearExceptionState.hpp"

namespace {
bool has_stashed_exception(Interpreter &interpreter)
{
	return interpreter.execution_frame()->stashed_exception_info().has_value();
}
}// namespace

void ClearExceptionState::execute(VirtualMachine &, Interpreter &interpreter) const
{
	ASSERT(has_stashed_exception(interpreter))

	interpreter.execution_frame()->clear_stashed_exception();
}

std::vector<uint8_t> ClearExceptionState::serialize() const { return { CLEAR_EXCEPTION_STATE }; }
