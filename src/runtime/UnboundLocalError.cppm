module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:unbound_local_error;
import :baseexception;
import :dict;
import :exception;
import :object;
import :string;
import :tuple;
import :value;
import py.memory;
import std;

export namespace py {
class PyType;

class UnboundLocalError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_unbound_local_error(std::string &&);

  private:
	UnboundLocalError(PyType *type);

	UnboundLocalError(PyTuple *args);

	static UnboundLocalError *create(PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
};

// Defined in UnboundLocalError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_unbound_local_error(std::string &&message);

template<typename... Args>
inline BaseException *unbound_local_error(const std::string &message, Args &&...args)
{
	return make_unbound_local_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
