#include "PyReversed.hpp"
#include "PyArgParser.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {
PyReversed::PyReversed(PyType *type) : PyBaseObject(type) {}

PyReversed::PyReversed(PyObject *sequence)
	: PyBaseObject(types::BuiltinTypes::the().reversed()), m_sequence(sequence)
{}

PyResult<PyObject *> PyReversed::create(PyObject *sequence)
{
	auto [reversed_method, lookup_result] =
		sequence->lookup_attribute(PyString::create("__reversed__").unwrap());
	if (reversed_method.is_err()) { return reversed_method; }
	if (lookup_result == LookupAttrResult::FOUND && reversed_method.unwrap() == py_none()) {
		return Err(type_error("'{}' object is not reversible", sequence->type()->name()));
	} else if (lookup_result == LookupAttrResult::FOUND) {
		return reversed_method.unwrap()->call(nullptr, nullptr);
	} else {
		TODO();
	}
}

PyResult<PyObject *> PyReversed::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::reversed());
	auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
		kwargs,
		"reversed",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 1>{});
	if (result.is_err()) return Err(result.unwrap_err());
	return PyReversed::create(std::get<0>(result.unwrap()));
}

PyResult<PyObject *> PyReversed::__iter__() const { TODO(); }
PyResult<PyObject *> PyReversed::__next__() { TODO(); }

PyType *PyReversed::static_type() const { return types::reversed(); }

void PyReversed::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_sequence) { visitor.visit(*m_sequence); }
}

namespace {

	std::once_flag reversed_flag;

	std::unique_ptr<TypePrototype> register_reversed()
	{
		return std::move(klass<PyReversed>("reversed").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyReversed::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(reversed_flag, []() { type = register_reversed(); });
		return std::move(type);
	};
}

}// namespace py
