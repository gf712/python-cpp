#include "Pop.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> Pop::execute(VirtualMachine &vm, Interpreter &) const
{
	auto value = vm.pop();
	if (!m_discard) { vm.reg(m_dst) = std::move(value); }
	// clean up the stack to remove dangling pointers
	vm.push(nullptr);
	vm.pop();
	return Ok(py_none());
}

std::vector<uint8_t> Pop::serialize() const
{
	return {
		POP,
		static_cast<uint8_t>(m_discard),
		m_dst,
	};
}