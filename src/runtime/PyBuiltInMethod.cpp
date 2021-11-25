#include "PyBuiltInMethod.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

PyBuiltInMethod::PyBuiltInMethod(std::string name,
	std::function<PyObject *(PyTuple *, PyDict *)> builtin_method,
	PyObject *self)
	: PyBaseObject(PyObjectType::PY_BUILTIN_METHOD, BuiltinTypes::the().builtin_method()),
	  m_name(std::move(name)), m_builtin_method(std::move(builtin_method)), m_self(self)
{}

void PyBuiltInMethod::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_self);
}

std::string PyBuiltInMethod::to_string() const
{
	return fmt::format("<built-in method '{}' of '{}' object at {}>",
		m_name,
		object_name(m_self->type()),
		static_cast<void *>(m_self));
}

PyObject *PyBuiltInMethod::__repr__() const { return PyString::create(to_string()); }

PyBuiltInMethod *PyBuiltInMethod::create(std::string name,
	std::function<PyObject *(PyTuple *, PyDict *)> builtin_method,
	PyObject *self)
{
	return VirtualMachine::the().heap().allocate<PyBuiltInMethod>(name, builtin_method, self);
}

PyType *PyBuiltInMethod::type_() const { return ::builtin_method(); }

namespace {

std::once_flag builtin_method_flag;

std::unique_ptr<TypePrototype> register_builtin_method()
{
	return std::move(klass<PyBuiltInMethod>("builtin_method").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyBuiltInMethod::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(builtin_method_flag, []() { type = ::register_builtin_method(); });
	return std::move(type);
}