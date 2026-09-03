module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> DeleteFast::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.stack_local(m_stack_index) = nullptr;
	return Ok(py_none());
}

std::vector<uint8_t> DeleteFast::serialize() const
{
	return {
		DELETE_FAST,
		m_stack_index,
	};
}
