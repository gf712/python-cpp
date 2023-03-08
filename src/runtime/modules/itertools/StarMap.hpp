#include "runtime/PyObject.hpp"

namespace py {
namespace itertools {
	class StarMap : public PyBaseObject
	{
		friend class ::Heap;

		PyObject *m_function{ nullptr };
		PyObject *m_iterator{ nullptr };

		StarMap(PyType *);
		StarMap(PyObject *function, PyObject *iterator);
		static PyResult<PyObject *> create(PyObject *function, PyObject *iterable);

	  public:
		static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

		PyResult<PyObject *> __iter__() const;
		PyResult<PyObject *> __next__();

		static PyType *register_type(PyModule *module);

		void visit_graph(Visitor &visitor) override;
	};
}// namespace itertools
}// namespace py
