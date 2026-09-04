module;
#include "core.hpp"

module py.runtime;
import py.types;


namespace py {

IndexError *IndexError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<IndexError>(args);
}

BaseException *make_index_error(std::string &&message)
{
	auto msg = PyString::create(std::move(message));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return IndexError::create(args_tuple.unwrap());
}

IndexError::IndexError(PyType *type) : LookupError(type, nullptr) {}

IndexError::IndexError(PyTuple *args) : LookupError(types::BuiltinTypes::the().index_error(), args)
{}

PyResult<PyObject *> IndexError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::index_error());
	ASSERT(!kwargs || kwargs->map().empty());
	if (auto result = IndexError::create(args)) {
		return Ok(static_cast<PyObject *>(result));
	} else {
		TODO();
	}
}

PyType *IndexError::class_type()
{
	ASSERT(types::index_error());
	return types::index_error();
}

PyType *IndexError::static_type() const
{
	ASSERT(types::index_error());
	return types::index_error();
}

namespace {

	std::once_flag index_error_flag;

	std::unique_ptr<TypePrototype> register_index_error()
	{
		return std::move(klass<IndexError>("IndexError", LookupError::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> IndexError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(index_error_flag, []() { type = register_index_error(); });
		return std::move(type);
	};
}

}// namespace py
