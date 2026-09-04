module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> StoreFast::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.stack_local(m_stack_index) = vm.reg(m_src);
	return py::Ok(vm.stack_local(m_stack_index));
}

std::vector<uint8_t> StoreFast::serialize() const
{
	return {
		STORE_FAST,
		m_stack_index,
		m_src,
	};
}