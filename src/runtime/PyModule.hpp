#pragma once

#include "PyObject.hpp"
#include "PyString.hpp"

class PyModule : public PyObject
{
  public:
	using MapType = std::unordered_map<PyString *, Value, ValueHash, ValueEqual>;

  private:
	friend class Heap;

	MapType m_module_definitions;
	PyString *m_module_name;

  public:
	PyModule(PyString *module_name)
		: PyObject(PyObjectType::PY_MODULE), m_module_name(std::move(module_name))
	{}

	void visit_graph(Visitor &visitor) override;

	PyObject *repr_impl(Interpreter &) const override
	{
		return PyString::create(fmt::format("<module '{}'>", m_module_name->to_string()));
	}

	std::string to_string() const override
	{
		return fmt::format("<module '{}'>", m_module_name->to_string());
	}

	const MapType &module_definitions() const { return m_module_definitions; }

	PyString *name() const { return m_module_name; }

	void insert(PyString *key, const Value &value)
	{
		m_module_definitions.insert_or_assign(key, value);
	}
};


template<> inline PyModule *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_MODULE) { return static_cast<PyModule *>(node); }
	return nullptr;
}


template<> inline const PyModule *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_MODULE) { return static_cast<const PyModule *>(node); }
	return nullptr;
}
