#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"

namespace py {
namespace itertools {
	class Count : public PyBaseObject
	{
		friend class ::Heap;

		Number m_start{ 0 };
		Number m_step{ 1 };
		Number m_current{ 0 };

		Count(PyType *type);

		Count();
		Count(Number start);
		Count(Number start, Number step);

		static PyResult<PyObject *> create();
		static PyResult<PyObject *> create(Number start);
		static PyResult<PyObject *> create(Number start, Number step);

	  public:
		static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

		PyResult<PyObject *> __iter__() const;
		PyResult<PyObject *> __next__();

		static PyType *register_type(PyModule *module);

		void visit_graph(Visitor &visitor) override;
	};
}// namespace itertools
}// namespace py
