#pragma once

#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/warnings/Warning.hpp"
#include "vm/VM.hpp"

namespace py {

class DeprecationWarning : public Warning
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *deprecation_warning(const std::string &message, Args &&...args);

  protected:
	DeprecationWarning(PyType *type);

	DeprecationWarning(PyType *, PyTuple *args);

  private:
	static PyResult<DeprecationWarning *> create(PyType *, PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
	static PyType *class_type();
};

template<typename... Args>
inline BaseException *deprecation_warning(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return DeprecationWarning::create(DeprecationWarning::class_type(), args_tuple.unwrap())
		.unwrap();
}

}// namespace py
