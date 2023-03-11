#include "StoreGlobal.hpp"

#include "interpreter/Interpreter.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyObject.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> StoreGlobal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &value = vm.reg(m_source);
	const auto &object_name = interpreter.execution_frame()->names(m_object_name);
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
