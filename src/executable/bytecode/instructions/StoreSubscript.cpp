#include "StoreSubscript.hpp"

py::PyResult StoreSubscript::execute(VirtualMachine &, Interpreter &) const
{
	TODO();
	return py::PyResult::Err(nullptr);
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
