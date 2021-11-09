#pragma once

#include "PyObject.hpp"
#include "PyString.hpp"

class PyModule : public PyObject
{
  public:
	using MapType = std::unordered_map<PyString *, Value, ValueHash, ValueEqual>;

  private:
	friend class Heap;

	MapType m_symbol_table;
	PyString *m_module_name;

	std::shared_ptr<Program> m_program;

  public:
	PyModule(PyString *module_name);

	static PyModule *create(PyString *);

	void visit_graph(Visitor &visitor) override;

	PyObject *repr_impl(Interpreter &) const override
	{
		return PyString::create(fmt::format("<module '{}'>", m_module_name->to_string()));
	}

	std::string to_string() const override
	{
		return fmt::format("<module '{}'>", m_module_name->to_string());
	}

	const MapType &symbol_table() const { return m_symbol_table; }

	PyString *name() const { return m_module_name; }

	void insert(PyString *key, const Value &value)
	{
		m_symbol_table.insert_or_assign(key, value);
		m_attributes.insert_or_assign(key->value(), PyObject::from(value));
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
