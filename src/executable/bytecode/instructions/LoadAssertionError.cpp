#include "LoadAssertionError.hpp"
#include "runtime/AssertionError.hpp"
#include "runtime/PyType.hpp"

using namespace py;

void LoadAssertionError::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.reg(m_assertion_location) = AssertionError::this_type();
}

std::vector<uint8_t> LoadAssertionError::serialize() const
{
	return {
		LOAD_ASSERTION_ERROR,
		m_assertion_location,
	};
}