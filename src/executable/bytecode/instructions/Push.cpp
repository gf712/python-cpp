#include "Push.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> Push::execute(VirtualMachine &vm, Interpreter &) const
{
	auto value = vm.reg(m_source);
	vm.push(value);
	return Ok(py_none());
}

std::vector<uint8_t> Push::serialize() const
{
	return {
		PUSH,
		m_source,
	};
}