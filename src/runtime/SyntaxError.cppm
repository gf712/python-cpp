module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:syntax_error;
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

class SyntaxError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_syntax_error(std::string &&);

  private:
	SyntaxError(PyType *type);

	SyntaxError(PyTuple *args);

	static SyntaxError *create(PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
};

// Defined in SyntaxError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_syntax_error(std::string &&message);

template<typename... Args>
inline BaseException *syntax_error(const std::string &message, Args &&...args)
{
	return make_syntax_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
