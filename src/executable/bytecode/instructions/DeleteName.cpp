#include "DeleteName.hpp"

using namespace py;

PyResult<Value>DeleteName::execute(VirtualMachine &vm, Interpreter &) const
{
	auto obj = vm.reg(m_name);
	TODO();
	return Err(nullptr);
}

std::vector<uint8_t> DeleteName::serialize() const
{
	return {
		DELETE_NAME,
		m_name,
	};
}
