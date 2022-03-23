#include "BinarySubscript.hpp"

void BinarySubscript::execute(VirtualMachine &vm, Interpreter &) const
{
	auto object = vm.reg(m_src);
	auto slice = vm.reg(m_index);
	(void)object;
	(void)slice;
	TODO();
}