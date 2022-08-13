#include "ListAppend.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> ListAppend::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &list = vm.reg(m_list);
	auto &value = vm.reg(m_value);

	ASSERT(std::holds_alternative<PyObject *>(list))

	auto *pylist = std::get<PyObject *>(list);
	ASSERT(pylist)
	ASSERT(as<PyList>(pylist))

	auto result = PyTuple::create(value);
	if (result.is_err()) { return Err(result.unwrap_err()); }

	if (auto r = as<PyList>(pylist)->append(result.unwrap(), nullptr); r.is_ok()) {
		return Ok(Value{ r.unwrap() });
	} else {
		return Err(r.unwrap_err());
	}
}

std::vector<uint8_t> ListAppend::serialize() const
{
	return {
		LIST_APPEND,
		m_list,
		m_value,
	};
}