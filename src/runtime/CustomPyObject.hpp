#pragma once

#include "PyObject.hpp"

struct CustomPyObjectContext
{
	const std::string name;
	const std::vector<std::pair<std::string, std::shared_ptr<PyObject>>> attributes;
	const std::vector<std::pair<std::string, std::shared_ptr<PyFunction>>> methods;
};

// a user defined PyObject
class CustomPyObject : public PyObject
{
  public:
	CustomPyObject(const CustomPyObjectContext &ctx, const std::shared_ptr<PyTuple> &)
		: PyObject(PyObjectType::PY_CUSTOM_TYPE)
	{
		// m_slots["__qualname__"] = [ctx](std::shared_ptr<PyTuple>, std::shared_ptr<PyDict>) {
		// 	return PyString::from(String{ ctx.name });
		// };
		for (const auto &[k, v] : m_slots) { spdlog::debug("Key: '{}' {}", k, (void *)v().get()); }

		for (const auto &[attr_name, attr_value] : ctx.attributes) {
			spdlog::debug("Adding attribute to class namespace: '{}' {}",
				attr_name,
				(void *)attr_value.get());
			put(attr_name, attr_value);
		}
	}

	std::string to_string() const override
	{
		// return fmt::format("CustomPyObject(__qualname__={})",
		// 	m_slots.at("__qualname__")(nullptr, nullptr)->to_string());
		return "test";
	}
};