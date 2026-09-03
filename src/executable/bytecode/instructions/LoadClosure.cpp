module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> LoadClosure::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto result = interpreter.execution_frame()->freevars()[m_source];
	vm.reg(m_destination) = result;
	return Ok(Value{ result });
}

std::vector<uint8_t> LoadClosure::serialize() const
{
	return {
		LOAD_CLOSURE,
		m_destination,
		m_source,
	};
}