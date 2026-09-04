module;
#include "memory/allocate.hpp"

export module py.runtime:key_error;
import :exception;
import py.memory;

export namespace py {
class PyType;

class KeyError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_key_error(std::string &&);

  private:
	KeyError(PyType *type);

	KeyError(PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	static PyResult<KeyError *> create(PyTuple *args);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
	static PyType *class_type();
};

// Defined in KeyError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_key_error(std::string &&message);

template<typename... Args>
inline BaseException *key_error(const std::string &message, Args &&...args)
{
	return make_key_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
