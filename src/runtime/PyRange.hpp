#pragma once

#include "PyObject.hpp"

class PyRange : public PyObject
{
	friend class Heap;

	int64_t m_start;
	int64_t m_stop;
	int64_t m_step;

  public:
	PyRange(int64_t stop) : PyRange(0, stop, 1) {}

	PyRange(int64_t start, int64_t stop) : PyRange(start, stop, 1) {}

	PyRange(int64_t start, int64_t stop, int64_t step)
		: PyObject(PyObjectType::PY_RANGE), m_start(start), m_stop(stop), m_step(step)
	{}

	std::string to_string() const override;

	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> iter_impl(Interpreter &interpreter) const override;

	int64_t start() const { return m_start; }
	int64_t stop() const { return m_stop; }
	int64_t step() const { return m_step; }
};


class PyRangeIterator : public PyObject
{
	friend class Heap;

	std::shared_ptr<const PyRange> m_pyrange;
	int64_t m_current_index;

  public:
	PyRangeIterator(std::shared_ptr<const PyRange> pyrange)
		: PyObject(PyObjectType::PY_RANGE_ITERATOR), m_pyrange(std::move(pyrange)), m_current_index(m_pyrange->start())
	{}

	std::string to_string() const override;

	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> next_impl(Interpreter &interpreter) override;
};
