#include "ClearExceptionState.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"

using namespace py;


PyResult<Value> ClearExceptionState::execute(VirtualMachine &, Interpreter &interpreter) const
{
	// FIXME: it would be safer to not have to have check if there is an active exception, and abort
	// 		  otherwise.
	//		  However, this is currently used to clear an exception raised from __exit__ during with
	//		  clause cleanup. And at compile time we can't tell if an exception will be raised. Note
	//		  that the raised exception is swallowed if __exit__ returns a truthy object
	// if (interpreter.execution_frame()->exception_info().has_value()) {
	// 	interpreter.execution_frame()->pop_exception();
	// 	if (interpreter.execution_frame()->exception_info().has_value()) {
	// 		spdlog::debug("next exception: {}",
	// 			interpreter.execution_frame()->exception_info()->exception->to_string());
	// 	}
	// }
	while (interpreter.execution_frame()->exception_info().has_value()) {
		interpreter.execution_frame()->pop_exception();
	}

	return Ok(Value{ py_none() });
}

std::vector<uint8_t> ClearExceptionState::serialize() const { return { CLEAR_EXCEPTION_STATE }; }
