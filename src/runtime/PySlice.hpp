#pragma once

#include "PyObject.hpp"

namespace py {

class PySlice : public PyBaseObject
{
	friend class ::Heap;

  public:
	PyObject *m_start{ nullptr };
	PyObject *m_stop{ nullptr };
	PyObject *m_step{ nullptr };

  private:
	PySlice(PyType *);

	PySlice();
	PySlice(PyObject *stop);
	PySlice(PyObject *start, PyObject *stop, PyObject *end);

  protected:
	void visit_graph(Visitor &) override;

  public:
	static PyResult<PySlice *> create(int64_t stop);
	static PyResult<PySlice *> create(int64_t start, int64_t stop, int64_t end);

	static PyResult<PySlice *> create(PyObject *stop);
	static PyResult<PySlice *> create(PyObject *start, PyObject *stop, PyObject *end);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __repr__() const;
	PyResult<int64_t> __hash__() const;
	PyResult<PyObject *> __eq__(const PyObject *obj) const;
	PyResult<PyObject *> __lt__(const PyObject *obj) const;

	PyResult<std::tuple<int64_t, int64_t, int64_t>> get_indices(int64_t length) const;

	PyResult<std::tuple<int64_t, int64_t, int64_t>> unpack() const;
	static std::tuple<int64_t, int64_t, int64_t>
		adjust_indices(int64_t start, int64_t stop, int64_t step, int64_t length);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
	std::string to_string() const override;
};

}// namespace py
