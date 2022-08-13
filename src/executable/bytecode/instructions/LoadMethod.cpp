#include "LoadMethod.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/AttributeError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyType.hpp"
#include "runtime/types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LoadMethod::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto this_value = vm.reg(m_value_source);
	const auto &method_name = interpreter.execution_frame()->names(m_method_name);
	spdlog::debug("This object: {}",
		std::visit(
			[](const auto &val) {
				auto obj = PyObject::from(val);
				ASSERT(obj.is_ok())
				return obj.unwrap()->to_string();
			},
			this_value));
	auto this_obj_ = PyObject::from(this_value);
	if (this_obj_.is_err()) { return Err(this_obj_.unwrap_err()); }
	auto *this_obj = this_obj_.unwrap();
	auto name = PyString::create(method_name);
	if (name.is_err()) { return Err(name.unwrap_err()); }
	auto maybe_method = this_obj->get_method(name.unwrap());
	if (maybe_method.is_err()) { return maybe_method; }
	if (auto *method = maybe_method.unwrap(); !method->is_callable()) {
		return Err(attribute_error("object '{}' is not callable", method->name()));
	} else {
		vm.reg(m_destination) = method;
	}
	return Ok(Value{ maybe_method.unwrap() });
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
