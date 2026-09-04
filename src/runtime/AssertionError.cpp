module;
#include "core.hpp"

module py.runtime;
import py.types;


namespace py {

PyResult<AssertionError *> AssertionError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto result = heap.allocate<AssertionError>(args);
	if (!result) { return Err(memory_error(sizeof(AssertionError))); }
	return Ok(result);
}

BaseException *make_assertion_error(std::string &&message)
{
	auto msg = PyString::create(std::move(message));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return AssertionError::create(args_tuple.unwrap()).unwrap();
}

AssertionError::AssertionError(PyType *type) : Exception(type->underlying_type(), nullptr) {}

AssertionError::AssertionError(PyTuple *args)
	: Exception(types::BuiltinTypes::the().assertion_error(), args)
{}

PyResult<PyObject *> AssertionError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::assertion_error());
	ASSERT(!kwargs || kwargs->map().empty());
	if (auto result = AssertionError::create(args); result.is_ok()) {
		return Ok(static_cast<PyObject *>(result.unwrap()));
	} else {
		return Err(result.unwrap_err());
	}
}

PyType *AssertionError::static_type() const
{
	ASSERT(types::assertion_error());
	return types::assertion_error();
}

PyType *AssertionError::this_type()
{
	ASSERT(types::assertion_error());
	return types::assertion_error();
}

std::string AssertionError::to_string() const { return what(); }

namespace {

	std::once_flag assertion_error_flag;

	std::unique_ptr<TypePrototype> register_assertion_error()
	{
		return std::move(klass<AssertionError>("AssertionError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> AssertionError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(assertion_error_flag, []() { type = register_assertion_error(); });
		return std::move(type);
	};
}
}// namespace py
