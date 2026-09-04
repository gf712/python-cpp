module;

#include "memory/allocate.hpp"

export module py.runtime:float_;
import :number;
import std;
import py.memory;

export namespace py {
class PyType;

class PyFloat final : public PyNumber
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	PyFloat(double);

	PyFloat(PyType *);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static PyResult<PyFloat *> create(double);
	PyType *static_type() const override;

	double as_f64() const;

	PyResult<PyObject *> __round__(PyObject *ndigits) const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
};

}// namespace py
