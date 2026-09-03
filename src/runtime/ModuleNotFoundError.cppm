module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:module_not_found_error;
import :import_error;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyString;
class PyDict;
class BaseException;
class PyNone;
class PyType;

class ModuleNotFoundError : public ImportError
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_module_not_found_error(std::string &&);

	PyObject *m_name{ nullptr };
	PyObject *m_path{ nullptr };

  private:
	ModuleNotFoundError(PyTuple *msg, PyObject *name, PyObject *path);

	ModuleNotFoundError(PyType *type);

	static ModuleNotFoundError *create(PyTuple *args, PyObject *name, PyObject *path);

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<std::int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyType *static_type() const override;
};

// Defined in ModuleNotFoundError.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_module_not_found_error(std::string &&message);

template<typename... Args>
inline BaseException *module_not_found_error(const std::string &message, Args &&...args)
{
	return make_module_not_found_error(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
