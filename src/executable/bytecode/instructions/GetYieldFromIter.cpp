module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> GetYieldFromIter::execute(VirtualMachine &vm, Interpreter &) const
{
	auto iterable_value = vm.reg(m_src);
	ASSERT(std::holds_alternative<PyObject *>(iterable_value));

	if (auto *generator = as<PyGenerator>(std::get<PyObject *>(iterable_value))) {
		vm.reg(m_dst) = generator;
		return Ok(generator);
	} else if (auto *coro = as<PyCoroutine>(std::get<PyObject *>(iterable_value))) {
		(void)coro;
		TODO();
	} else {
		[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
		return std::get<PyObject *>(iterable_value)->iter().and_then([this, &vm](PyObject *obj) {
			vm.reg(m_dst) = obj;
			return Ok(obj);
		});
	}
}

std::vector<uint8_t> GetYieldFromIter::serialize() const
{
	return {
		GET_YIELD_FROM_ITER,
		m_dst,
		m_src,
	};
}
