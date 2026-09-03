module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> GetAwaitable::execute(VirtualMachine &, Interpreter &) const
{
	TODO();
	return Err(nullptr);
}

std::vector<uint8_t> GetAwaitable::serialize() const
{
	return {
		GET_AWAITABLE,
		m_dst,
		m_src,
	};
}
