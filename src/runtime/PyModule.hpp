#pragma once

#include "PyObject.hpp"
#include "PyString.hpp"

class PyModule : public PyObject
{
  public:
	using MapType = std::unordered_map<std::shared_ptr<PyString>, Value, ValueHash, ValueEqual>;

  private:
	friend class Heap;

	MapType m_module_definitions;
	std::shared_ptr<PyString> m_module_name;

  public:
	PyModule(std::shared_ptr<PyString> module_name)
		: PyObject(PyObjectType::PY_MODULE), m_module_name(std::move(module_name))
	{}

	std::shared_ptr<PyObject> repr_impl(Interpreter &) const override
	{
		return PyString::create(fmt::format("<module '{}'>", m_module_name->to_string()));
	}

	std::string to_string() const override
	{
		return fmt::format("<module '{}'>", m_module_name->to_string());
	}

	const MapType &module_definitions() const { return m_module_definitions; }

	void insert(const std::shared_ptr<PyString> &key, const Value &value)
	{
		m_module_definitions.insert_or_assign(key, value);
	}
};


template<> inline std::shared_ptr<PyModule> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_MODULE) {
		return std::static_pointer_cast<PyModule>(node);
	}
	return nullptr;
}