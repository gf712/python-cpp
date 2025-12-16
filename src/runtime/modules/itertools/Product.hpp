#pragma once

#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"

namespace py {
namespace itertools {
	class Product : public PyBaseObject
	{
		friend class ::Heap;

		PyList *m_pools{ nullptr };
		size_t m_repeat;
		std::vector<std::vector<Value>> m_result;
		size_t m_iteration_count{ 0 };

		Product(PyType *);
		Product(PyList *pools, size_t repeat);
		static PyResult<PyObject *> create(PyObject *iterable, std::optional<size_t> length);

	  public:
		static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

		PyResult<PyObject *> __iter__() const;
		PyResult<PyObject *> __next__();

		static PyType *register_type(PyModule *module);

		void visit_graph(Visitor &visitor) override;
	};
}// namespace itertools
}// namespace py
