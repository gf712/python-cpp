#include "ClearTopCleanup.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

using namespace py;


PyResult<Value> ClearTopCleanup::execute(VirtualMachine &vm, Interpreter &) const
{
	ASSERT(vm.state().cleanup.size() > 1);
	vm.state().cleanup.pop();

	return Ok(Value{ py_none() });
}

std::vector<uint8_t> ClearTopCleanup::serialize() const { return { CLEAR_TOP_CLEANUP }; }
