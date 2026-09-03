module;
#include "core.hpp"

module py.runtime;
import py.types;


namespace py {

Exception *Exception::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<Exception>(args);
}

BaseException *make_exception(std::string &&message)
{
	auto msg = PyString::create(std::move(message));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return Exception::create(args_tuple.unwrap());
}

Exception::Exception(PyType *t) : BaseException(t->underlying_type(), nullptr) {}

Exception::Exception(PyType *t, PyTuple *args) : BaseException(t, args) {}

Exception::Exception(PyTuple *args) : BaseException(types::BuiltinTypes::the().exception(), args) {}

Exception::Exception(const TypePrototype &type, PyTuple *args) : BaseException(type, args) {}

PyResult<PyObject *> Exception::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::exception());
	ASSERT(!kwargs || kwargs->map().empty());
	return Ok(Exception::create(args));
}

PyType *Exception::static_type() const
{
	ASSERT(types::exception());
	return types::exception();
}

PyType *Exception::class_type()
{
	ASSERT(types::exception());
	return types::exception();
}

namespace {

	std::once_flag exception_flag;

	std::unique_ptr<TypePrototype> register_exception()
	{
		return std::move(klass<Exception>("Exception", BaseException::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> Exception::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(exception_flag, []() { type = register_exception(); });
		return std::move(type);
	};
}
}// namespace py
