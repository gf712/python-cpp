#pragma once

#include "PyObject.hpp"

namespace py {

class PyRange : public PyBaseObject
{
	friend class ::Heap;

	int64_t m_start;
	int64_t m_stop;
	int64_t m_step;

  public:
	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;

	int64_t start() const { return m_start; }
	int64_t stop() const { return m_stop; }
	int64_t step() const { return m_step; }

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;

  private:
	PyRange(int64_t stop);
	PyRange(int64_t start, int64_t stop);
	PyRange(int64_t start, int64_t stop, int64_t step);
};


class PyRangeIterator : public PyBaseObject
{
	friend class ::Heap;

	const PyRange &m_pyrange;
	int64_t m_current_index;

  public:
	PyRangeIterator(const PyRange &);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	void visit_graph(Visitor &) override;

	PyType *type() const override;
};

}// namespace py