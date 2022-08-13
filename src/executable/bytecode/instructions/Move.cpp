#include "Move.hpp"
#include "vm/VM.hpp"

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