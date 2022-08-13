#pragma once

#include "PyObject.hpp"

namespace py {

class PyProperty : public PyBaseObject
{
  public:
	PyObject *m_getter{ nullptr };
	PyObject *m_setter{ nullptr };
	PyObject *m_deleter{ nullptr };

  private:
	PyObject *m_property_name{ nullptr };

	friend class ::Heap;

	PyProperty(PyObject *fget, PyObject *fset, PyObject *fdel, PyObject *);

  public:
	static PyResult<PyProperty *>
		create(PyObject *fget, PyObject *fset, PyObject *fdel, PyObject *);

	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __get__(PyObject *instance, PyObject *owner) const;
	PyResult<std::monostate> __set__(PyObject *obj, PyObject *value);

	PyResult<PyObject *> getter(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> setter(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> deleter(PyTuple *args, PyDict *kwargs) const;

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
};

}// namespace py