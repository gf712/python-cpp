module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> ListExtend::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &list = vm.reg(m_list);
	auto &value = vm.reg(m_value);

	ASSERT(std::holds_alternative<PyObject *>(list));

	auto *pylist = std::get<PyObject *>(list);
	ASSERT(pylist);
	ASSERT(as<PyList>(pylist));

	return PyObject::from(value).and_then([pylist](PyObject *iterable) {
		// extend() walks the iterator protocol, which may run Python __iter__ /
		// __next__ and clobber r0.
		[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;

		return as<PyList>(pylist)->extend(iterable);
	});
}

std::vector<uint8_t> ListExtend::serialize() const
{
	return {
		LIST_EXTEND,
		m_list,
		m_value,
	};
}
