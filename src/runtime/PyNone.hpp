#pragma once

#include "PyObject.hpp"

namespace py {

class PyNone : public PyBaseObject
{
	friend class ::Heap;
	friend PyObject *py_none();

	bool m_value;

  public:
	std::string to_string() const override;

	PyResult<PyObject *> __add__(const PyObject *obj) const;
	PyResult<PyObject *> __repr__() const;

	bool value() const { return m_value; }

	void visit_graph(Visitor &) override {}

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

  private:
	static PyNone *create();
	PyNone();
};

PyObject *py_none();

}// namespace py