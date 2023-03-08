#pragma once

#include "PyObject.hpp"
#include "PyString.hpp"

namespace py {

class PyModule : public PyBaseObject
{
  public:
	PyString *m_module_name{ nullptr };
	PyObject *m_doc{ nullptr };
	PyObject *m_package{ nullptr };
	PyObject *m_loader{ nullptr };
	PyObject *m_spec{ nullptr };
	PyDict *m_dict{ nullptr };

  private:
	friend class ::Heap;
	friend class VM;

	std::shared_ptr<Program> m_program;

	PyModule(PyType *);

  public:
	PyModule(PyDict *symbol_table, PyString *module_name, PyObject *doc);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	static PyResult<PyModule *> create(PyDict *symbol_table, PyString *module_name, PyObject *doc);

	void visit_graph(Visitor &visitor) override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __getattribute__(PyObject *attribute) const;

	std::string to_string() const override;

	PyDict *symbol_table() const { return m_attributes; }
	void add_symbol(PyString *key, const Value &value);

	PyString *name() const { return m_module_name; }

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	void set_program(std::shared_ptr<Program> program);

	const std::shared_ptr<Program> &program() const;
};

}// namespace py
