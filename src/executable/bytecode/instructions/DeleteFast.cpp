#include "DeleteFast.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

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
