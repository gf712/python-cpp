module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> DeleteGlobal::execute(VirtualMachine &, Interpreter &interpreter) const
{
	auto name = interpreter.execution_frame()->consts(m_name);
	auto name_str = PyObject::from(name);
	if (name_str.is_err()) return name_str;
	[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
	return interpreter.execution_frame()
		->globals()
		->delete_item(name_str.unwrap())
		.and_then([](auto) { return Ok(py_none()); });
}

std::vector<uint8_t> DeleteGlobal::serialize() const
{
	return {
		DELETE_GLOBAL,
		m_name,
	};
}
