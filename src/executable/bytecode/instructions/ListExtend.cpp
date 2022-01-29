#include "ListExtend.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyTuple.hpp"


void ListExtend::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &list = vm.reg(m_list);
	auto &value = vm.reg(m_value);

	ASSERT(std::holds_alternative<PyObject *>(list))

	auto *pylist = std::get<PyObject *>(list);
	ASSERT(pylist)
	ASSERT(as<PyList>(pylist))

	as<PyList>(pylist)->extend(PyTuple::create(value), nullptr);
}