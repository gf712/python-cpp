#include "TypeError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

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
