#include "PyBoundMethod.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

PyBoundMethod::PyBoundMethod(PyObject *self, PyFunction *method)
	: PyBaseObject(BuiltinTypes::the().bound_method()), m_self(self), m_method(method)
{}

PyResult<PyBoundMethod *> PyBoundMethod::create(PyObject *self, PyFunction *method)
{
	auto *result = VirtualMachine::the().heap().allocate<PyBoundMethod>(self, method);
	if (!result) { return Err(memory_error(sizeof(PyBoundMethod))); }
	return Ok(result);
}

void PyBoundMethod::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_self);
	visitor.visit(*m_method);
}

std::string PyBoundMethod::to_string() const
{
	auto qualname_str = PyString::create("__qualname__");
	if (qualname_str.is_err()) { TODO(); }
	auto self_qualname = m_self->getattribute(qualname_str.unwrap());
	if (self_qualname.is_err()) { TODO(); }
	return fmt::format(
		"<bound method '{}' of '{}'>", m_method->name(), self_qualname.unwrap()->to_string());
}

PyResult<PyObject *> PyBoundMethod::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyBoundMethod::__call__(PyTuple *args, PyDict *kwargs)
{
	// first create new args tuple -> (self, *args)
	std::vector<Value> new_args_vector;
	new_args_vector.reserve(args->size() + 1);
	new_args_vector.push_back(m_self);
	for (const auto &arg : args->elements()) { new_args_vector.push_back(arg); }
	auto args_ = PyTuple::create(new_args_vector);
	if (args_.is_err()) { return args_; }
	return m_method->call(args_.unwrap(), kwargs);
}

PyType *PyBoundMethod::type() const { return bound_method(); }

namespace {

std::once_flag bound_method_flag;

std::unique_ptr<TypePrototype> register_bound_method()
{
	return std::move(klass<PyBoundMethod>("bound_method").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyBoundMethod::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(bound_method_flag, []() { type = ::register_bound_method(); });
	return std::move(type);
}

template<> PyBoundMethod *py::as(PyObject *node)
{
	if (node->type() == bound_method()) { return static_cast<PyBoundMethod *>(node); }
	return nullptr;
}

template<> const PyBoundMethod *py::as(const PyObject *node)
{
	if (node->type() == bound_method()) { return static_cast<const PyBoundMethod *>(node); }
	return nullptr;
}