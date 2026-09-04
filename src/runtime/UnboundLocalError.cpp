module;
#include "core.hpp"

module py.runtime;
import py.types;


namespace py {

UnboundLocalError *UnboundLocalError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<UnboundLocalError>(args);
}

BaseException *make_unbound_local_error(std::string &&message)
{
	auto msg = PyString::create(std::move(message));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return UnboundLocalError::create(args_tuple.unwrap());
}

UnboundLocalError::UnboundLocalError(PyType *type) : Exception(type) {}

UnboundLocalError::UnboundLocalError(PyTuple *args)
	: Exception(types::BuiltinTypes::the().unbound_local_error(), args)
{}

PyResult<PyObject *> UnboundLocalError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::unbound_local_error());
	ASSERT(!kwargs || kwargs->map().empty());
	return Ok(UnboundLocalError::create(args));
}

PyType *UnboundLocalError::static_type() const
{
	ASSERT(types::unbound_local_error());
	return types::unbound_local_error();
}

namespace {

	std::once_flag unbound_local_error_flag;

	std::unique_ptr<TypePrototype> register_unbound_local_error()
	{
		return std::move(
			klass<UnboundLocalError>("UnboundLocalError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> UnboundLocalError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(unbound_local_error_flag, []() { type = register_unbound_local_error(); });
		return std::move(type);
	};
}

}// namespace py
