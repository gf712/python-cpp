#include "BinarySubscript.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> BinarySubscript::execute(VirtualMachine &vm, Interpreter &) const
{
	auto object_value = vm.reg(m_src);
	auto object = PyObject::from(object_value);
	if (object.is_err()) return object;

	auto subscript_value = vm.reg(m_index);
	auto subscript = PyObject::from(subscript_value);
	if (subscript.is_err()) return subscript;

	return object.and_then([&subscript](PyObject *obj) { return obj->getitem(subscript.unwrap()); })
		.and_then([&vm, this](PyObject *value) {
			vm.reg(m_dst) = value;
			return Ok(value);
		});
}

std::vector<uint8_t> BinarySubscript::serialize() const
{
	return {
		BINARY_SUBSCRIPT,
		m_dst,
		m_src,
		m_index,
	};
}
