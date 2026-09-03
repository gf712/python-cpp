module;
#include "core.hpp"

module py.runtime;
import py.types;


namespace py {

TypeError *TypeError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<TypeError>(args);
}

BaseException *make_type_error(std::string &&message)
{
	auto msg = PyString::create(std::move(message));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return TypeError::create(args_tuple.unwrap());
}

TypeError::TypeError(PyType *type) : Exception(type) {}

TypeError::TypeError(PyTuple *args) : Exception(types::BuiltinTypes::the().type_error(), args) {}

PyResult<PyObject *> TypeError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::type_error());
	ASSERT(!kwargs || kwargs->map().empty());
	return Ok(TypeError::create(args));
}

PyType *TypeError::static_type() const
{
	ASSERT(types::type_error());
	return types::type_error();
}

namespace {

	std::once_flag type_error_flag;

	std::unique_ptr<TypePrototype> register_type_error()
	{
		return std::move(klass<TypeError>("TypeError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> TypeError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(type_error_flag, []() { type = register_type_error(); });
		return std::move(type);
	};
}

}// namespace py
