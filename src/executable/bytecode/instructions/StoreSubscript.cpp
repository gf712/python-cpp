#include "StoreSubscript.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> StoreSubscript::execute(VirtualMachine &vm, Interpreter &) const
{
	auto object_value = vm.reg(m_obj);
	auto object = PyObject::from(object_value);
	if (object.is_err()) return object;

	auto subscript_value = vm.reg(m_slice);
	auto subscript = PyObject::from(subscript_value);
	if (subscript.is_err()) return subscript;

	auto value_value = vm.reg(m_src);
	auto value = PyObject::from(value_value);
	if (value.is_err()) return value;

	[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
	return object.unwrap()->setitem(subscript.unwrap(), value.unwrap()).and_then([](auto) {
		return Ok(py_none());
	});
}

std::vector<uint8_t> StoreSubscript::serialize() const
{
	return {
		STORE_SUBSCRIPT,
		m_obj,
		m_slice,
		m_src,
	};
}
