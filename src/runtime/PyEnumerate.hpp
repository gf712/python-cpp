#pragma once

#include "PyObject.hpp"

namespace py {
class PyEnumerate : public PyBaseObject
{
	friend class ::Heap;

	int64_t m_current_index{ 0 };
	PyObject *m_iterator{ nullptr };

  private:
	PyEnumerate(PyType *);

	PyEnumerate(int64_t current_index, PyObject *iterator);

  public:
	static PyResult<PyObject *> create(int64_t current_index, PyObject *iterable);
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};
}// namespace py
