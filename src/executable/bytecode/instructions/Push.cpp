module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


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