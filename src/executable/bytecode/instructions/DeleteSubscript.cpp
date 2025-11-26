#include "DeleteSubscript.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyObject.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> DeleteSubscript::execute(VirtualMachine &vm, Interpreter &) const
{
	auto value_ = vm.reg(m_value);
	auto index_ = vm.reg(m_index);

	auto value = PyObject::from(value_);
	if (value.is_err()) return value;
	auto index = PyObject::from(index_);
	if (index.is_err()) return index;

	[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
	return value.unwrap()->delete_item(index.unwrap()).and_then([](auto) { return Ok(py_none()); });
}

std::vector<uint8_t> DeleteSubscript::serialize() const
{
	return {
		DELETE_SUBSCRIPT,
		m_value,
		m_index,
	};
}
