#pragma once

#include "PyNumber.hpp"

namespace py {

class PyInteger : public Interface<PyNumber, PyInteger>
{
	friend class ::Heap;

	PyInteger(BigIntType);

  protected:
	PyInteger(PyType *);

	PyInteger(TypePrototype &, BigIntType);

  public:
	static PyResult<PyInteger *> create(int64_t);

	static PyResult<PyInteger *> create(BigIntType);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<int64_t> __hash__() const;

	PyResult<PyObject *> __and__(PyObject *obj);
	PyResult<PyObject *> __or__(PyObject *obj);

	PyResult<PyObject *> __lshift__(const PyObject *other) const;
	PyResult<PyObject *> __rshift__(const PyObject *other) const;

	PyResult<PyObject *> bit_length() const;
	PyResult<PyObject *> bit_count() const;

	PyResult<PyObject *> to_bytes(PyTuple *args, PyDict *kwargs) const;

	static PyResult<PyObject *> from_bytes(PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	int64_t as_i64() const;
	size_t as_size_t() const;
	BigIntType as_big_int() const;
};

}// namespace py
