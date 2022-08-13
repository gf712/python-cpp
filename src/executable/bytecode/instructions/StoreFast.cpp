#include "StoreFast.hpp"
#include "vm/VM.hpp"

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