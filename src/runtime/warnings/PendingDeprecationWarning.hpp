#pragma once

#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/warnings/Warning.hpp"
#include "vm/VM.hpp"

namespace py {

class PendingDeprecationWarning : public Warning
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *pending_deprecation_warning(const std::string &message, Args &&...args);

  protected:
	PendingDeprecationWarning(PyType *type);

	PendingDeprecationWarning(PyType *, PyTuple *args);

  private:
	static PyResult<PendingDeprecationWarning *> create(PyType *, PyTuple *args);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
	static PyType *class_type();
};

template<typename... Args>
inline BaseException *pending_deprecation_warning(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	return PendingDeprecationWarning::create(
		PendingDeprecationWarning::class_type(), args_tuple.unwrap())
		.unwrap();
}

}// namespace py
