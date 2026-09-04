module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:name_error;
import :exception;
import py.memory;

export namespace py {
class PyType;

class NameError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_name_error(std::string &&);

  private:
	NameError(PyType *type);

	NameError(PyTuple *args);

	static NameError *create(PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
};

// Defined in NameError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_name_error(std::string &&message);

template<typename... Args>
inline BaseException *name_error(const std::string &message, Args &&...args)
{
	return make_name_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
