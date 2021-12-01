#include "PyMethodWrapper.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

PyMethodWrapper::PyMethodWrapper(PyString *name,
	PyType *underlying_type,
	std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function,
	std::vector<PyObject *> &&captures)
	: PyBaseObject(PyObjectType::PY_METHOD_WRAPPER, BuiltinTypes::the().method_wrapper()),
	  m_name(std::move(name)), m_underlying_type(underlying_type),
	  m_method_descriptor(std::move(function)), m_captures(std::move(captures))
{}


void PyMethodWrapper::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	visitor.visit(*m_underlying_type);
	for (auto *capture : m_captures) { visitor.visit(*capture); }
}

std::string PyMethodWrapper::to_string() const
{
	return fmt::format(
		"<method '{}' of '{}' objects>", m_name->to_string(), m_underlying_type->name());
}

PyObject *PyMethodWrapper::__repr__() const { return PyString::create(to_string()); }

PyObject *PyMethodWrapper::__call__(PyTuple *args, PyDict *kwargs)
{
	// split args tuple -> (args[0], args[1:])
	// since args[0] is self (hopefully)
	std::vector<Value> new_args_vector;
	new_args_vector.reserve(args->size() - 1);
	PyObject *self = PyObject::from(args->elements()[0]);
	for (size_t i = 1; i < args->size(); ++i) { new_args_vector.push_back(args->elements()[i]); }
	args = PyTuple::create(new_args_vector);
	return m_method_descriptor(self, args, kwargs);
}


PyType *PyMethodWrapper::type_() const { return method_wrapper(); }

namespace {

std::once_flag method_wrapper_flag;

std::unique_ptr<TypePrototype> register_method_wrapper()
{
	return std::move(klass<PyMethodWrapper>("method_wrapper").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyMethodWrapper::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(method_wrapper_flag, []() { type = ::register_method_wrapper(); });
	return std::move(type);
}