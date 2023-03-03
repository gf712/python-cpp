#include "runtime/PyObject.hpp"

namespace py {
namespace itertools {
	class Chain : public PyBaseObject
	{
		friend class ::Heap;

		PyObject *m_iterable_objects_iterator{ nullptr };
		PyObject *m_current_iterator{ nullptr };

		Chain(PyObject *iterable_objects_iterator);
		static PyResult<PyObject *> create(PyTuple *iterable_objects);
		static PyResult<PyObject *> create(PyObject *iterable_objects);

	  public:
		static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

		PyResult<PyObject *> __iter__() const;
		PyResult<PyObject *> __next__();

		static PyResult<PyObject *> from_iterable(PyType *, PyTuple *args, PyDict *kwargs);

		static PyType *register_type(PyModule *module);

		void visit_graph(Visitor &visitor) override;
	};
}// namespace itertools
}// namespace py
