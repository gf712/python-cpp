module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:import_error;
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

class ImportError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_import_error(std::string &&);

  public:
	PyObject *m_name{ nullptr };

  protected:
	ImportError(PyType *);

	ImportError(PyType *, PyTuple *args);

	ImportError(TypePrototype &, PyTuple *args);

  private:
	ImportError(PyTuple *args);

	static ImportError *create(PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<std::int32_t> __init__(PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	static PyType *class_type();

	PyType *static_type() const override;

	void visit_graph(Visitor &) override;
};

// Defined in ImportError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_import_error(std::string &&message);

template<typename... Args>
inline BaseException *import_error(const std::string &message, Args &&...args)
{
	return make_import_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
