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
		spdlog::debug("Building object from context");
		for (const auto &[k, v] : m_attributes) {
			spdlog::debug("Key: '{}' {}", k, (void *)v.get());
		}

		for (const auto &[attr_name, attr_value] : ctx.attributes) {
			spdlog::debug("Adding attribute to class namespace: '{}' {}",
				attr_name,
				(void *)attr_value.get());
			if (!update_slot_if_special(attr_name, attr_value)) { put(attr_name, attr_value); }
		}
	}

	std::string to_string() const override { return fmt::format("CustomPyObject"); }

  private:
	bool update_slot_if_special(const std::string &name, std::shared_ptr<PyObject> value)
	{
		if (!name.starts_with("__")) { return false; }

		if (name == "__repr__") {
			auto pyfunc = as<PyFunction>(value);
			ASSERT(pyfunc)
			m_slots.repr = std::move(pyfunc);
		} else {
			spdlog::debug("{} is not a special name, skipping", name);
			return false;
		}

		return true;
	}
};