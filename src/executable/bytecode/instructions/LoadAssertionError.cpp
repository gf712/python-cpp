module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;

// Names Instruction, so it follows the import.
import py.types;
import std;


using namespace py;

PyResult<Value> LoadAssertionError::execute(VirtualMachine &vm, Interpreter &) const
{
	auto *result = types::assertion_error();
	// TODO: return a meaningful error. If this is nullptr then it is a serious internal error...
	if (!result) {
		TODO();
		return Err(nullptr);
	}
	vm.reg(m_assertion_location) = result;
	return Ok(Value{ result });
}

std::vector<uint8_t> LoadAssertionError::serialize() const
{
	return {
		LOAD_ASSERTION_ERROR,
		m_assertion_location,
	};
}