#pragma once

#include "forward.hpp"
#include "runtime/PyObject.hpp"
#include <cstddef>
#include <cstdint>
#include <deque>

namespace py {
namespace collections {
	class Deque : public PyBaseObject
	{
		friend class ::Heap;

		std::deque<Value> m_deque;
		std::optional<size_t> m_maxlength;

		Deque(PyType *);
		Deque(std::deque<Value>);

	  public:
		static PyResult<PyObject *>
			create(PyType *, std::deque<Value>, std::optional<size_t> maxlength);

		static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *);
		PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

		PyResult<PyObject *> append(PyObject *x);
		PyResult<PyObject *> appendleft(PyObject *x);
		PyResult<PyObject *> clear();
		PyResult<PyObject *> copy();
		PyResult<PyObject *> count(PyObject *x);
		PyResult<PyObject *> extend(PyObject *iterable);
		PyResult<PyObject *> extendleft(PyObject *iterable);
		PyResult<PyObject *> pop();
		PyResult<PyObject *> popleft();
		PyResult<PyObject *> remove(PyObject *value);
		PyResult<PyObject *> reverse();
		PyResult<PyObject *> rotate(PyObject *n);

		PyResult<PyObject *> __repr__() const;

		PyResult<size_t> __len__() const;
		PyResult<PyObject *> __getitem__(int64_t) const;

		static PyType *register_type(PyModule *module);

		void visit_graph(Visitor &visitor) override;

	  private:
		void push_back(Value);
		void push_front(Value);
	};
}// namespace collections
}// namespace py
