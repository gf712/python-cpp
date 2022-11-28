#include "PyNumber.hpp"

namespace py {

class PyFloat final : public PyNumber
{
	friend class ::Heap;

	PyFloat(double);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static PyResult<PyFloat *> create(double);
	PyType *type() const override;

	double as_f64() const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
};

}// namespace py
