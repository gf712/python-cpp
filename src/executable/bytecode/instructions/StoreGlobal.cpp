module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> StoreGlobal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &value = vm.reg(m_source);
	const auto &object_name = interpreter.execution_frame()->names(m_object_name);
	[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
	return interpreter.execution_frame()->put_global(object_name, value).and_then([](auto) {
		return Ok(Value{ py_none() });
	});
}

std::vector<uint8_t> StoreGlobal::serialize() const
{
	return {
		STORE_GLOBAL,
		m_object_name,
		m_source,
	};
}
