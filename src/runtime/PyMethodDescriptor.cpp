#include "PyMethodDescriptor.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

PyMethodDescriptor::PyMethodDescriptor(PyString *name,
	PyType *underlying_type,
	std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function,
	std::vector<PyObject *> &&captures)
	: PyBaseObject(BuiltinTypes::the().method_wrapper()), m_name(std::move(name)),
	  m_underlying_type(underlying_type), m_method_descriptor(std::move(function)),
	  m_captures(std::move(captures))
{}


void PyMethodDescriptor::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	visitor.visit(*m_underlying_type);
	for (auto *capture : m_captures) { visitor.visit(*capture); }
}

std::string PyMethodDescriptor::to_string() const
{
	return fmt::format(
		"<method '{}' of '{}' objects>", m_name->to_string(), m_underlying_type->name());
}

PyObject *PyMethodDescriptor::__repr__() const { return PyString::create(to_string()); }

PyObject *PyMethodDescriptor::__call__(PyTuple *args, PyDict *kwargs)
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

PyObject *PyMethodDescriptor::__get__(PyObject *instance, PyObject * /*owner*/) const
{
	if (!instance) { return const_cast<PyMethodDescriptor *>(this); }
	if (instance->type() != m_underlying_type) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("descriptor '{}' for '{}' objects "
					   "doesn't apply to a '{}' object",
				m_name->value(),
				m_underlying_type->underlying_type().__name__,
				instance->type()->underlying_type().__name__));
		return nullptr;
	}
	return PyNativeFunction::create(
		m_name->value(),
		[this, instance](PyTuple *args, PyDict *kwargs) {
			std::vector<Value> new_args_vector;
			new_args_vector.reserve(args->size() + 1);
			new_args_vector.push_back(instance);
			new_args_vector.insert(
				new_args_vector.end(), args->elements().begin(), args->elements().end());
			args = PyTuple::create(new_args_vector);
			return const_cast<PyMethodDescriptor *>(this)->__call__(args, kwargs);
		},
		const_cast<PyMethodDescriptor *>(this),
		instance);
}


PyType *PyMethodDescriptor::type() const { return method_wrapper(); }

namespace {

std::once_flag method_wrapper_flag;

std::unique_ptr<TypePrototype> register_method_wrapper()
{
	return std::move(klass<PyMethodDescriptor>("method_descriptor").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyMethodDescriptor::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(method_wrapper_flag, []() { type = ::register_method_wrapper(); });
	return std::move(type);
}

template<> PyMethodDescriptor *py::as(PyObject *obj)
{
	if (obj->type() == method_wrapper()) { return static_cast<PyMethodDescriptor *>(obj); }
	return nullptr;
}

template<> const PyMethodDescriptor *py::as(const PyObject *obj)
{
	if (obj->type() == method_wrapper()) { return static_cast<const PyMethodDescriptor *>(obj); }
	return nullptr;
}