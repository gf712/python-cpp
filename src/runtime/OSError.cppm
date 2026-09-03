module;
#include "memory/allocate.hpp"

export module py.runtime:o_s_error;
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

class OSError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_os_error(std::string &&);

  private:
	OSError(PyTuple *args);

  protected:
	OSError(PyType *);
	OSError(PyType *, PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	static PyResult<OSError *> create(PyTuple *args);
	static PyResult<OSError *> create(PyType *, PyTuple *args);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

	static PyType *class_type();
};

// Defined in OSError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_os_error(std::string &&message);

template<typename... Args>
inline BaseException *os_error(const std::string &message, Args &&...args)
{
	return make_os_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
