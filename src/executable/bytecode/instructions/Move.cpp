#include "Move.hpp"

py::PyResult Move::execute(VirtualMachine &vm, Interpreter &) const
{
	auto result = vm.reg(m_source);
	vm.reg(m_destination) = result;
	return py::PyResult::Ok(result);
}

std::vector<uint8_t> Move::serialize() const
{
	return {
		MOVE,
		m_destination,
		m_source,
	};
}