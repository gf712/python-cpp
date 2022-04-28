#include "ListToTuple.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyTuple.hpp"

using namespace py;

PyResult ListToTuple::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &list = vm.reg(m_list);

	ASSERT(std::holds_alternative<PyObject *>(list))

	auto *pylist = std::get<PyObject *>(list);
	ASSERT(as<PyList>(pylist))

	auto result = PyTuple::create(as<PyList>(pylist)->elements());
	if (result.is_ok()) { vm.reg(m_tuple) = result.unwrap(); }
	return result;
}

std::vector<uint8_t> ListToTuple::serialize() const
{
	return {
		LIST_TO_TUPLE,
		m_tuple,
		m_list,
	};
}