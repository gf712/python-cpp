#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class RuntimeError : public Exception
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *runtime_error(const std::string &message, Args &&...args);

  private:
	RuntimeError(PyType *);

	RuntimeError(PyTuple *args);

	static PyResult<RuntimeError *> create(PyTuple *args);

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;
};

template<typename... Args>
inline BaseException *runtime_error(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok())
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok())
	auto obj = RuntimeError::create(args_tuple.unwrap());
	ASSERT(obj.is_ok())
	return obj.unwrap();
}

}// namespace py
