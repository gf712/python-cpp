#include "runtime/PyObject.hpp"

namespace py {
namespace itertools {
	class ISlice : public PyBaseObject
	{
		friend class ::Heap;

		PyObject *m_iterator{ nullptr };
		BigIntType m_start{ 0 };
		std::optional<BigIntType> m_stop;
		BigIntType m_step{ 1 };
		std::optional<BigIntType> m_counter;

		ISlice(PyObject *iterator,
			BigIntType start,
			std::optional<BigIntType> stop,
			BigIntType step);
		static PyResult<PyObject *>
			create(PyObject *iterable, PyObject *start, PyObject *stop, PyObject *step);

	  public:
		static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

		PyResult<PyObject *> __iter__() const;
		PyResult<PyObject *> __next__();

		static PyType *register_type(PyModule *module);

		void visit_graph(Visitor &visitor) override;
	};
}// namespace itertools
}// namespace py
