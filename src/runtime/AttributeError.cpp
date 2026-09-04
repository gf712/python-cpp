module;
#include "core.hpp"

module py.runtime;
import py.types;


namespace py {

AttributeError *AttributeError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<AttributeError>(args);
}

BaseException *make_attribute_error(std::string &&message)
{
	auto msg = PyString::create(std::move(message));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return AttributeError::create(args_tuple.unwrap());
}

AttributeError::AttributeError(PyType *type) : Exception(type) {}

AttributeError::AttributeError(PyTuple *args)
	: Exception(types::BuiltinTypes::the().attribute_error(), args)
{}

PyResult<PyObject *> AttributeError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::attribute_error());
	ASSERT(!kwargs || kwargs->map().empty());
	return Ok(AttributeError::create(args));
}

PyType *AttributeError::static_type() const
{
	ASSERT(types::attribute_error());
	return types::attribute_error();
}

PyType *AttributeError::class_type()
{
	ASSERT(types::attribute_error());
	return types::attribute_error();
}

namespace {

	std::once_flag attribute_error_flag;

	std::unique_ptr<TypePrototype> register_attribute_error()
	{
		return std::move(klass<AttributeError>("AttributeError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> AttributeError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(attribute_error_flag, []() { type = register_attribute_error(); });
		return std::move(type);
	};
}

}// namespace py
