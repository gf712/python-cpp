#pragma once

#include "Warning.hpp"

namespace py {

class ImportWarning : public Warning
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *import_warning(const std::string &message, Args &&...args);

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

template<typename... Args>
inline BaseException *import_warning(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return ImportWarning::create(ImportWarning::class_type(), args_tuple.unwrap()).unwrap();
}

}// namespace py
