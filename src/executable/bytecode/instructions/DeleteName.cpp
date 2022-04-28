#include "DeleteName.hpp"

using namespace py;

PyResult DeleteName::execute(VirtualMachine &vm, Interpreter &) const
{
	auto obj = vm.reg(m_name);
	TODO();
	return PyResult::Err(nullptr);
}

std::vector<uint8_t> DeleteName::serialize() const
{
	return {
		DELETE_NAME,
		m_name,
	};
}
