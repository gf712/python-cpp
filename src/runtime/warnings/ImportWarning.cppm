module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:import_warning;
import :warning;
import :baseexception;
import :dict;
import :object;
import :string;
import :tuple;
import :value;
import py.memory;
import std;

export namespace py {
class PyType;

class ImportWarning : public Warning
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *make_import_warning(std::string &&);

  private:
	ImportWarning(PyType *type);

	ImportWarning(PyType *type, PyTuple *args);

	static PyResult<ImportWarning *> create(PyType *type, PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
	static PyType *class_type();
};

// Defined in warnings/ImportWarning.cpp. Keeping the PyString/PyTuple construction and the
// heap allocation out of the interface means this partition no longer has
// to materialise those types just to declare one exception class.
BaseException *make_import_warning(std::string &&message);

template<typename... Args>
inline BaseException *import_warning(const std::string &message, Args &&...args)
{
	return make_import_warning(std::vformat(message, std::make_format_args(args...)));
}

}// namespace py
