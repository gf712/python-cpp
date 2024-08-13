#pragma once

#include "runtime/PyDict.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"

namespace py {
namespace collections {
	class DefaultDict : public PyDict
	{
		friend class ::Heap;

		PyObject *m_default_factory;

		DefaultDict(PyType *);
		DefaultDict(PyObject *default_factory);

	  public:
        PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

		static PyType *register_type(PyModule *module);

		void visit_graph(Visitor &visitor) override;

        PyResult<PyObject*> __missing__(PyObject *key);
	};
}// namespace collection
}// namespace py
