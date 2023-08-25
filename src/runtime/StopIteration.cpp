#include "StopIteration.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> StopIteration *as(PyObject *obj)
{
	ASSERT(types::stop_iteration());
	if (obj->type() == types::stop_iteration()) { return static_cast<StopIteration *>(obj); }
	return nullptr;
}

template<> const StopIteration *as(const PyObject *obj)
{
	ASSERT(types::stop_iteration());
	if (obj->type() == types::stop_iteration()) { return static_cast<const StopIteration *>(obj); }
	return nullptr;
}

StopIteration::StopIteration(PyType *type) : Exception(type) {}

StopIteration::StopIteration(PyTuple *args)
	: Exception(types::BuiltinTypes::the().stop_iteration(), args)
{}

PyResult<PyObject *> StopIteration::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::stop_iteration());
	ASSERT(!kwargs || kwargs->map().empty())
	return Ok(StopIteration::create(args));
}

PyType *StopIteration::static_type() const
{
	ASSERT(types::stop_iteration())
	return types::stop_iteration();
}

PyType *StopIteration::class_type()
{
	ASSERT(types::stop_iteration());
	return types::stop_iteration();
}


namespace {

	std::once_flag stop_iteration_flag;

	std::unique_ptr<TypePrototype> register_stop_iteration()
	{
		return std::move(klass<StopIteration>("StopIteration", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> StopIteration::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(stop_iteration_flag, []() { type = register_stop_iteration(); });
		return std::move(type);
	};
}
}// namespace py
