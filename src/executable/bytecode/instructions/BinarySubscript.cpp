#include "BinarySubscript.hpp"

using namespace py;

PyResult<Value> BinarySubscript::execute(VirtualMachine &vm, Interpreter &) const
{
	auto object = vm.reg(m_src);
	auto slice = vm.reg(m_index);
	(void)object;
	(void)slice;
	TODO();
	return Err(nullptr);
}

std::vector<uint8_t> BinarySubscript::serialize() const
{
	return {
		BINARY_SUBSCRIPT,
		m_dst,
		m_src,
		m_index,
	};
}
