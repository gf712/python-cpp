#include "LoadException.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyFrame.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LoadException::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	// Bind the currently-active exception *instance* (used by `except ... as name`).
	const auto exception_info = interpreter.execution_frame()->exception_info();
	ASSERT(exception_info.has_value());
	auto *exception = static_cast<PyObject *>(exception_info->exception);
	ASSERT(exception);
	vm.reg(m_destination) = exception;
	return Ok(Value{ exception });
}

std::vector<uint8_t> LoadException::serialize() const
{
	return {
		LOAD_EXCEPTION,
		m_destination,
	};
}
