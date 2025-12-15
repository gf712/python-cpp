#include "SetUpdate.hpp"
#include "runtime/PySet.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> SetUpdate::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &set = vm.reg(m_set);
	auto &iterable = vm.reg(m_iterable);

	ASSERT(std::holds_alternative<PyObject *>(set));

	auto *pyset = std::get<PyObject *>(set);
	ASSERT(pyset);
	ASSERT(as<PySet>(pyset));
	auto iterable_obj = PyObject::from(iterable);
	if (iterable_obj.is_err()) { return Err(iterable_obj.unwrap_err()); }

	if (auto r = as<PySet>(pyset)->update(iterable_obj.unwrap()); r.is_ok()) {
		return Ok(Value{ r.unwrap() });
	} else {
		return Err(r.unwrap_err());
	}
}

std::vector<uint8_t> SetUpdate::serialize() const
{
	return {
		SET_UPDATE,
		m_set,
		m_iterable,
	};
}
