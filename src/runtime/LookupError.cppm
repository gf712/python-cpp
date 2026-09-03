module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:lookup_error;
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

class LookupError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_lookup_error(std::string &&);

  protected:
	LookupError(PyType *);

	LookupError(PyType *, PyTuple *msg);

	LookupError(TypePrototype &, PyTuple *msg);

  private:
	LookupError(PyTuple *msg);

	static LookupError *create(PyTuple *args);

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyType *static_type() const override;
	static PyType *class_type();
};

// Defined in LookupError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_lookup_error(std::string &&message);

template<typename... Args>
inline BaseException *lookup_error(const std::string &message, Args &&...args)
{
	return make_lookup_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
