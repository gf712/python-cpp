#include "runtime/PyObject.hpp"

namespace py {
namespace itertools {
	class Repeat : public PyBaseObject
	{
		friend class ::Heap;

		PyObject *m_object{ nullptr };
		std::optional<BigIntType> m_times_remaining;

		Repeat(PyType *type);
		Repeat(PyObject *object);
		Repeat(PyObject *object, BigIntType times);
		static PyResult<PyObject *> create(PyObject *object);
		static PyResult<PyObject *> create(PyObject *object, BigIntType times);

	  public:
		static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

		PyResult<PyObject *> __iter__() const;
		PyResult<PyObject *> __next__();

		static PyType *register_type(PyModule *module);

		void visit_graph(Visitor &visitor) override;
	};
}// namespace itertools
}// namespace py
