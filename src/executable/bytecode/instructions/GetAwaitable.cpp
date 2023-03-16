#include "GetAwaitable.hpp"
#include "runtime/PyObject.hpp"

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
