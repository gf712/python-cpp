#pragma once

#include "PyObject.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/Value.hpp"

namespace py {

class PyMemoryView : public PyBaseObject
{
	friend class ::Heap;

	struct ManagedBuffer
	{
		PyBuffer m_main_view;
	};

	std::shared_ptr<ManagedBuffer> m_managed_buffer;// the original view
	PyBuffer m_view;// our view

	PyMemoryView(PyType *);
	PyMemoryView(PyBuffer);

  public:
	static PyResult<PyObject *> create(PyObject *object);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;

	PyResult<size_t> __len__() const;

	PyResult<PyObject *> cast(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> tolist();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	void visit_graph(Visitor &) override;
	std::string to_string() const override;

	size_t itemsize() const { return m_view.itemsize; }

  private:
	static PyResult<PyBuffer> create_view(PyBuffer &main_view);
};

}// namespace py
