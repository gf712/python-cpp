module;
#include "core.hpp"

module py.runtime;
import py.types;


namespace py {

SyntaxError *SyntaxError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<SyntaxError>(args);
}

BaseException *make_syntax_error(std::string &&message)
{
	auto msg = PyString::create(std::move(message));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return SyntaxError::create(args_tuple.unwrap());
}

SyntaxError::SyntaxError(PyType *type) : Exception(type) {}

SyntaxError::SyntaxError(PyTuple *args) : Exception(types::BuiltinTypes::the().syntax_error(), args)
{}

PyResult<PyObject *> SyntaxError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::syntax_error());
	ASSERT(!kwargs || kwargs->map().empty());
	return Ok(SyntaxError::create(args));
}

PyType *SyntaxError::static_type() const
{
	ASSERT(types::syntax_error());
	return types::syntax_error();
}

namespace {

	std::once_flag syntax_error_flag;

	std::unique_ptr<TypePrototype> register_syntax_error()
	{
		return std::move(klass<SyntaxError>("SyntaxError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> SyntaxError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(syntax_error_flag, []() { type = register_syntax_error(); });
		return std::move(type);
	};
}

}// namespace py
