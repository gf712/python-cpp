module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> LoadConst::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(vm.registers().has_value());
	ASSERT(vm.registers()->get().size() > m_destination);
	auto result = interpreter.execution_frame()->consts(m_static_value_index);
	vm.reg(m_destination) = result;
	return py::Ok(result);
}

std::vector<uint8_t> LoadConst::serialize() const
{
	ASSERT(m_static_value_index < std::numeric_limits<uint8_t>::max());
	return {
		LOAD_CONST,
		m_destination,
		static_cast<u_int8_t>(m_static_value_index),
	};
}