module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


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
