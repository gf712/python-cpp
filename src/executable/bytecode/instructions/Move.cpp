#include "Move.hpp"

void Move::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.reg(m_destination) = vm.reg(m_source);
}

std::vector<uint8_t> Move::serialize() const
{
	return {
		MOVE,
		m_destination,
		m_source,
	};
}