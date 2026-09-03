module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> LoadMethod::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto this_value = vm.reg(m_value_source);
	const auto &method_name = interpreter.execution_frame()->names(m_method_name);
	auto this_obj_ = PyObject::from(this_value);
	if (this_obj_.is_err()) { return Err(this_obj_.unwrap_err()); }
	auto *this_obj = this_obj_.unwrap();
	auto name = PyString::create(method_name);
	return name
		.and_then([this_obj](PyString *method_name) {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return this_obj->get_method(method_name);
		})
		.and_then([&vm, this](PyObject *method_obj) {
			vm.reg(m_destination) = method_obj;
			return Ok(method_obj);
		});
}

std::vector<uint8_t> LoadMethod::serialize() const
{
	return {
		LOAD_METHOD,
		m_destination,
		m_value_source,
		m_method_name,
	};
}
