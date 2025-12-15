#include "ListAppend.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> ListAppend::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &list = vm.reg(m_list);
	auto &value = vm.reg(m_value);

	ASSERT(std::holds_alternative<PyObject *>(list));

	auto *pylist = std::get<PyObject *>(list);
	ASSERT(pylist);
	ASSERT(as<PyList>(pylist));

	return PyObject::from(value).and_then(
		[pylist](PyObject *element) { return as<PyList>(pylist)->append(element); });
}

std::vector<uint8_t> ListAppend::serialize() const
{
	return {
		LIST_APPEND,
		m_list,
		m_value,
	};
}