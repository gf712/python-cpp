#include "LoadAssertionError.hpp"
#include "runtime/AssertionError.hpp"
#include "runtime/PyType.hpp"

using namespace py;

PyResult LoadAssertionError::execute(VirtualMachine &vm, Interpreter &) const
{
	auto *result = AssertionError::this_type();
	// TODO: return a meaningful error. If this is nullptr then it is a serious internal error...
	if (!result) {
		TODO();
		return PyResult::Err(nullptr);
	}
	vm.reg(m_assertion_location) = result;
	return PyResult::Ok(result);
}

std::vector<uint8_t> LoadAssertionError::serialize() const
{
	return {
		LOAD_ASSERTION_ERROR,
		m_assertion_location,
	};
}