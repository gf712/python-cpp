#include "ListExtend.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyTuple.hpp"

using namespace py;

PyResult ListExtend::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &list = vm.reg(m_list);
	auto &value = vm.reg(m_value);

	ASSERT(std::holds_alternative<PyObject *>(list))

	auto *pylist = std::get<PyObject *>(list);
	ASSERT(pylist)
	ASSERT(as<PyList>(pylist))

	auto result = PyTuple::create(value);
	if (result.is_err()) { return result; }

	return as<PyList>(pylist)->extend(result.unwrap_as<PyTuple>(), nullptr);
}

std::vector<uint8_t> ListExtend::serialize() const
{
	return {
		LIST_EXTEND,
		m_list,
		m_value,
	};
}