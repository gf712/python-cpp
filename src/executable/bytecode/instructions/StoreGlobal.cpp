#include "StoreGlobal.hpp"

#include "runtime/PyNone.hpp"
#include "runtime/PyObject.hpp"

using namespace py;

PyResult<Value> StoreGlobal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &value = vm.reg(m_source);
	interpreter.execution_frame()->put_global(m_object_name, value);
	return Ok(Value{ py_none() });
}

std::vector<uint8_t> StoreGlobal::serialize() const
{
	TODO();
	return {
		STORE_GLOBAL,
		m_source,
	};
}
