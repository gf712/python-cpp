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

	static PyResult __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult __repr__() const;
	PyResult __iter__() const;

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
	friend class ::Heap;

	const PyRange &m_pyrange;
	int64_t m_current_index;

  public:
	PyRangeIterator(const PyRange &);

	std::string to_string() const override;

	PyResult __repr__() const;
	PyResult __next__();

	static std::unique_ptr<TypePrototype> register_type();

	void visit_graph(Visitor &) override;

	PyType *type() const override;
};

}// namespace py