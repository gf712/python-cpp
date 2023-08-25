#include "PyEnumerate.hpp"
#include "MemoryError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {
PyEnumerate::PyEnumerate(PyType *type) : PyBaseObject(type) {}

PyEnumerate::PyEnumerate(int64_t current_index, PyObject *iterator)
	: PyBaseObject(types::BuiltinTypes::the().enumerate()), m_current_index(current_index),
	  m_iterator(iterator)
{}

PyResult<PyObject *> PyEnumerate::create(int64_t current_index, PyObject *iterable)
{
	if (auto iterator = iterable->iter(); iterator.is_err()) {
		return Err(type_error("'{}' object is not iterable", iterable->type()->name()));
	} else {
		auto *result =
			VirtualMachine::the().heap().allocate<PyEnumerate>(current_index, iterator.unwrap());
		if (!result) { return Err(memory_error(sizeof(PyEnumerate))); }
		return Ok(result);
	}
}

PyResult<PyObject *> PyEnumerate::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::enumerate());
	auto result = PyArgsParser<PyObject *, int64_t>::unpack_tuple(args,
		kwargs,
		"enumerate",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 2>{},
		0 /* start */);
	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [iterable, start] = result.unwrap();

	return PyEnumerate::create(start, iterable);
}

PyResult<PyObject *> PyEnumerate::__iter__() const { return Ok(const_cast<PyEnumerate *>(this)); }

PyResult<PyObject *> PyEnumerate::__next__()
{
	auto value = m_iterator->next();
	if (value.is_err()) { return value; }
	return PyTuple::create(Number{ m_current_index++ }, value.unwrap());
}

PyType *PyEnumerate::static_type() const { return types::enumerate(); }

void PyEnumerate::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_iterator) { visitor.visit(*m_iterator); }
}

namespace {

	std::once_flag enumerate_flag;

	std::unique_ptr<TypePrototype> register_enumerate()
	{
		return std::move(klass<PyEnumerate>("enumerate").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyEnumerate::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(enumerate_flag, []() { type = register_enumerate(); });
		return std::move(type);
	};
}

}// namespace py
