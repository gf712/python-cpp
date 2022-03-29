#include "StoreSubscript.hpp"

void StoreSubscript::execute(VirtualMachine &, Interpreter &) const { TODO(); }

std::vector<uint8_t> StoreSubscript::serialize() const
{
	return {
		STORE_SUBSCRIPT,
		m_obj,
		m_slice,
		m_src,
	};
}
