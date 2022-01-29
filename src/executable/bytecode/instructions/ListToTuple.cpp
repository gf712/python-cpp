#include "ListToTuple.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyTuple.hpp"


void ListToTuple::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &list = vm.reg(m_list);

	ASSERT(std::holds_alternative<PyObject *>(list))

	auto *pylist = std::get<PyObject *>(list);
	ASSERT(as<PyList>(pylist))

	vm.reg(m_tuple) = PyTuple::create(as<PyList>(pylist)->elements());
}