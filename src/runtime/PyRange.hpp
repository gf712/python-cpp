#pragma once

#include "PyObject.hpp"

namespace py {

class PyRange : public PyBaseObject
{
	friend class ::Heap;

	const BigIntType m_start{ 0 };
	const BigIntType m_stop;
	const BigIntType m_step{ 1 };

	PyRange(PyType *);

  public:
	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;

	const BigIntType &start() const { return m_start; }
	const BigIntType &stop() const { return m_stop; }
	const BigIntType &step() const { return m_step; }

	PyResult<PyObject *> __reversed__() const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	PyRange(BigIntType start, BigIntType stop, BigIntType step);
	PyRange(PyInteger *stop);
	PyRange(PyInteger *start, PyInteger *stop);
	PyRange(PyInteger *start, PyInteger *stop, PyInteger *step);
};


class PyRangeIterator : public PyBaseObject
{
	friend class ::Heap;

	const PyRange &m_pyrange;
	BigIntType m_current_index;

  public:
	PyRangeIterator(const PyRange &);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();
	PyResult<PyObject *> __iter__() const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	void visit_graph(Visitor &) override;

	PyType *static_type() const override;
};

}// namespace py
