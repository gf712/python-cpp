#include "LoadFast.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LoadFast::execute(VirtualMachine &vm, Interpreter &) const
{
	auto result = vm.stack_local(m_stack_index);
	vm.reg(m_destination) = result;
	return py::Ok(result);
}

std::vector<uint8_t> LoadFast::serialize() const
{
	return {
		LOAD_FAST,
		m_destination,
		m_stack_index,
	};
}