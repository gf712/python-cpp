#pragma once

#include "PyObject.hpp"
#include <cstddef>

namespace py {

class PyIterator : public PyBaseObject
{
	friend class ::Heap;

	size_t m_index{ 0 };
	PyObject *m_iterator{ nullptr };

	PyIterator(PyObject *iterator);

  public:
	static PyResult<PyIterator *> create(PyObject *iterator);

	void visit_graph(Visitor &) override;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();

	PyResult<size_t> __len__() const;

    PyType *static_type() const override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
};

}// namespace py