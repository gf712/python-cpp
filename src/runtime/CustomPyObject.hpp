#pragma once

#include "PyObject.hpp"

struct CustomPyObjectContext
{
	const std::string name;
};

// a user defined PyObject
class CustomPyObject : public PyObject
{
	std::unordered_map<std::string,
		std::function<std::shared_ptr<PyObject>(std::shared_ptr<PyObject>)>>
		m_attributes;

  public:
	CustomPyObject(const CustomPyObjectContext &ctx, const std::shared_ptr<PyTuple> &)
		: PyObject(PyObjectType::PY_CUSTOM_TYPE)
	{
		m_attributes["__qualname__"] = [ctx](std::shared_ptr<PyObject>) {
			return PyString::from(String{ ctx.name });
		};
	}

	std::string to_string() const
	{
		return fmt::format("CustomPyObject(__qualname__={}",
			m_attributes.at("__qualname__")(nullptr)->to_string());
	}
};