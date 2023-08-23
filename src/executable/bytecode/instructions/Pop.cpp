#include "Pop.hpp"
#include "vm/VM.hpp"
#include "runtime/PyNone.hpp"

using namespace py;

PyResult<Value> Pop::execute(VirtualMachine &vm, Interpreter &) const {
	vm.pop();
	vm.push(nullptr);
	vm.pop();
	return Ok(py_none()); 
}

std::vector<uint8_t> Pop::serialize() const
{
	return {
		POP,
	};
}