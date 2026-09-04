module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;


PyResult<Value> ClearTopCleanup::execute(VirtualMachine &vm, Interpreter &) const
{
	ASSERT(vm.state().cleanup.size() > 1);
	vm.state().cleanup.pop();

	return Ok(Value{ py_none() });
}

std::vector<uint8_t> ClearTopCleanup::serialize() const { return { CLEAR_TOP_CLEANUP }; }
