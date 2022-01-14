#pragma once

#include "PyObject.hpp"
#include "PyString.hpp"

namespace py {

class PyModule : public PyBaseObject
{
  public:
	using MapType = std::unordered_map<PyString *, Value, ValueHash, ValueEqual>;

  private:
	friend class ::Heap;

	MapType m_symbol_table;
	PyString *m_module_name;

	std::shared_ptr<Program> m_program;

  public:
	PyModule(PyString *module_name);

	static PyModule *create(PyString *);

	void visit_graph(Visitor &visitor) override;

	PyObject *__repr__() const;

	std::string to_string() const override;

	const MapType &symbol_table() const { return m_symbol_table; }

	PyString *name() const { return m_module_name; }

	void insert(PyString *key, const Value &value);

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py