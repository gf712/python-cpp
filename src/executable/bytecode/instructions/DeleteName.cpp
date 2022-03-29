#include "DeleteName.hpp"

void DeleteName::execute(VirtualMachine &vm, Interpreter &) const
{
	auto obj = vm.reg(m_name);
	TODO();
}

std::vector<uint8_t> DeleteName::serialize() const
{
	return {
		DELETE_NAME,
		m_name,
	};
}
