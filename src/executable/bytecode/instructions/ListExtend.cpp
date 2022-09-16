#include "ListExtend.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> ListExtend::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &list = vm.reg(m_list);
	auto &value = vm.reg(m_value);

	ASSERT(std::holds_alternative<PyObject *>(list))

	auto *pylist = std::get<PyObject *>(list);
	ASSERT(pylist)
	ASSERT(as<PyList>(pylist))

	return PyObject::from(value).and_then(
		[pylist](PyObject *iterable) { return as<PyList>(pylist)->extend(iterable); });
}

std::vector<uint8_t> ListExtend::serialize() const
{
	return {
		LIST_EXTEND,
		m_list,
		m_value,
	};
}