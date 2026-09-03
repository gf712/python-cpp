module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> Move::execute(VirtualMachine &vm, Interpreter &) const
{
	auto result = vm.reg(m_source);
	vm.reg(m_destination) = result;
	return Ok(result);
}

std::vector<uint8_t> Move::serialize() const
{
	return {
		MOVE,
		m_destination,
		m_source,
	};
}