#include "CustomPyObject.hpp"
#include "PyDict.hpp"
#include "PyString.hpp"


CustomPyObject::CustomPyObject(const CustomPyObjectContext &ctx, const std::shared_ptr<PyTuple> &)
	: PyObject(PyObjectType::PY_CUSTOM_TYPE)
{
	// m_slots["__qualname__"] = [ctx](std::shared_ptr<PyTuple>, std::shared_ptr<PyDict>) {
	// 	return PyString::from(String{ ctx.name });
	// };
	spdlog::debug("Building object from context");
	for (const auto &[k, v] : m_attributes) { spdlog::debug("Key: '{}' {}", k, (void *)v.get()); }

	for (const auto &[attr_name, attr_value] : ctx.attributes->map()) {
		auto attr_name_str = as<PyString>(PyObject::from(attr_name));
		ASSERT(attr_name_str);
		auto attr_value_obj = PyObject::from(attr_value);
		spdlog::debug("Adding attribute to class namespace: '{}' {}",
			attr_name_str->to_string(),
			(void *)attr_value_obj.get());
		if (!update_slot_if_special(attr_name_str->to_string(), attr_value_obj)) {
			put(attr_name_str->to_string(), attr_value_obj);
		}
	}
}