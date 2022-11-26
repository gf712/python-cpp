#include "SetAdd.hpp"
#include "runtime/PySet.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> SetAdd::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &set = vm.reg(m_set);
	auto &value = vm.reg(m_value);

	ASSERT(std::holds_alternative<PyObject *>(set))

	auto *pyset = std::get<PyObject *>(set);
	ASSERT(pyset)
	ASSERT(as<PySet>(pyset))
	auto value_obj = PyObject::from(value);
	if (value_obj.is_err()) { return Err(value_obj.unwrap_err()); }

	if (auto r = as<PySet>(pyset)->add(value_obj.unwrap()); r.is_ok()) {
		return Ok(Value{ r.unwrap() });
	} else {
		return Err(r.unwrap_err());
	}
}

std::vector<uint8_t> SetAdd::serialize() const
{
	return {
		SET_ADD,
		m_set,
		m_value,
	};
}
