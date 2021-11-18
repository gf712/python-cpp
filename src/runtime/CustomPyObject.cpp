#include "CustomPyObject.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "vm/VM.hpp"


CustomPyObject::CustomPyObject(const CustomPyObjectContext &ctx, const PyTuple *)
	: PyBaseObject(PyObjectType::PY_CUSTOM_TYPE)
{
	// m_slots["__qualname__"] = [ctx](std::shared_ptr<PyTuple>, std::shared_ptr<PyDict>) {
	// 	return PyString::from(String{ ctx.name });
	// };
	spdlog::debug("Building object from context");
	for (const auto &[k, v] : m_attributes) { spdlog::debug("Key: '{}' {}", k, (void *)v); }

	for (const auto &[attr_name, attr_value] : ctx.attributes->map()) {
		auto attr_name_str = as<PyString>(PyObject::from(attr_name));
		ASSERT(attr_name_str);
		auto attr_value_obj = PyObject::from(attr_value);
		spdlog::debug("Adding attribute to class namespace: '{}' {}",
			attr_name_str->to_string(),
			(void *)attr_value_obj);
		if (auto method = as<PyFunction>(attr_value_obj)) {
			// a function becomes a bound method, ie. first argument is always self
			auto *bound_method = VirtualMachine::the().heap().allocate<PyBoundMethod>(this, method);
			put(attr_name_str->to_string(), bound_method);
		} else if (!update_slot_if_special(attr_name_str->to_string(), attr_value_obj)) {
			put(attr_name_str->to_string(), attr_value_obj);
		}
	}
}


bool CustomPyObject::update_slot_if_special(const std::string &name, PyObject *value)
{
	if (!name.starts_with("__")) { return false; }

	if (name == "__repr__") {
		auto pyfunc = as<PyFunction>(value);
		ASSERT(pyfunc)
		m_slots->repr = std::move(pyfunc);
	} else {
		spdlog::debug("{} is not a special name, skipping", name);
		return false;
	}

	return true;
}