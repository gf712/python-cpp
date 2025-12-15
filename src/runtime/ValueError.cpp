#include "ValueError.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> ValueError *as(PyObject *obj)
{
	ASSERT(types::value_error());
	if (obj->type() == types::value_error()) { return static_cast<ValueError *>(obj); }
	return nullptr;
}

template<> const ValueError *as(const PyObject *obj)
{
	ASSERT(types::value_error());
	if (obj->type() == types::value_error()) { return static_cast<const ValueError *>(obj); }
	return nullptr;
}

ValueError::ValueError(PyType *type) : Exception(type) {}

ValueError::ValueError(PyTuple *args) : Exception(types::BuiltinTypes::the().value_error(), args) {}

PyResult<ValueError *> ValueError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto result = heap.allocate<ValueError>(args);
	if (!result) { return Err(memory_error(sizeof(ValueError))); }
	return Ok(result);
}

PyResult<PyObject *> ValueError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::value_error());
	ASSERT(!kwargs || kwargs->map().empty());
	return ValueError::create(args);
}

PyType *ValueError::static_type() const
{
	ASSERT(types::value_error());
	return types::value_error();
}

PyType *ValueError::class_type()
{
	ASSERT(types::value_error());
	return types::value_error();
}

namespace {

	std::once_flag value_error_flag;

	std::unique_ptr<TypePrototype> register_value_error()
	{
		return std::move(klass<ValueError>("ValueError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> ValueError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(value_error_flag, []() { type = register_value_error(); });
		return std::move(type);
	};
}

}// namespace py
