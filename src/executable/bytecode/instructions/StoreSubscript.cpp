#include "StoreSubscript.hpp"

using namespace py;

PyResult<Value> StoreSubscript::execute(VirtualMachine &, Interpreter &) const
{
	TODO();
	return Err(nullptr);
}

std::vector<uint8_t> StoreSubscript::serialize() const
{
	return {
		STORE_SUBSCRIPT,
		m_obj,
		m_slice,
		m_src,
	};
}
