#pragma once

#include "PyObject.hpp"

class PyRange : public PyBaseObject
{
	friend class Heap;

	int64_t m_start;
	int64_t m_stop;
	int64_t m_step;

  public:
	std::string to_string() const override;

	static PyObject *__new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyObject *__repr__() const;
	PyObject *__iter__() const;

	int64_t start() const { return m_start; }
	int64_t stop() const { return m_stop; }
	int64_t step() const { return m_step; }

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

  private:
	PyRange(int64_t stop);
	PyRange(int64_t start, int64_t stop);
	PyRange(int64_t start, int64_t stop, int64_t step);
};


class PyRangeIterator : public PyBaseObject
{
	friend class Heap;

	const PyRange &m_pyrange;
	int64_t m_current_index;

  public:
	PyRangeIterator(const PyRange &);

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *__next__();

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};
