module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:memory_error;
import :exception;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyString;
class PyDict;
class BaseException;
class PyType;

class MemoryError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *memory_error(std::size_t failed_allocation_size);

  private:
	MemoryError(PyType *type);

	MemoryError(PyTuple *args);

	static PyResult<MemoryError *> create(PyTuple *args);

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyType *static_type() const override;

	static PyType *this_type();

	std::string to_string() const override;
};

template<> MemoryError *as(PyObject *obj);
template<> const MemoryError *as(const PyObject *obj);

// Defined in MemoryError.cpp: keeping the PyString/PyTuple construction out
// of the interface means this partition no longer materialises those types.
BaseException *memory_error(std::size_t failed_allocation_size);

}// namespace py
