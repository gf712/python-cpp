#pragma once

#include "PyObject.hpp"

struct CustomPyObjectContext
{
	const std::string name;
	const std::shared_ptr<PyDict> attributes;
};

// a user defined PyObject
class CustomPyObject : public PyObject
{
  public:
	CustomPyObject(const CustomPyObjectContext &ctx, const std::shared_ptr<PyTuple> &);

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