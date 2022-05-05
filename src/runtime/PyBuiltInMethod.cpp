#include "PyBuiltInMethod.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyBuiltInMethod *as(PyObject *node)
{
	if (node->type() == builtin_method()) { return static_cast<PyBuiltInMethod *>(node); }
	return nullptr;
}

template<> const PyBuiltInMethod *as(const PyObject *node)
{
	if (node->type() == builtin_method()) { return static_cast<const PyBuiltInMethod *>(node); }
	return nullptr;
}

PyBuiltInMethod::PyBuiltInMethod(std::string name, FunctionType &&builtin_method, PyObject *self)
	: PyBaseObject(BuiltinTypes::the().builtin_method()), m_name(std::move(name)),
	  m_builtin_method(std::move(builtin_method)), m_self(self)
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
		m_self->type()->name(),
		static_cast<void *>(m_self));
}

PyResult<PyObject *> PyBuiltInMethod::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyBuiltInMethod::__call__(PyTuple *args, PyDict *kwargs)
{
	return m_builtin_method(args, kwargs);
}

PyResult<PyBuiltInMethod *>
	PyBuiltInMethod::create(std::string name, FunctionType &&builtin_method, PyObject *self)
{
	auto *obj = VirtualMachine::the().heap().allocate<PyBuiltInMethod>(
		name, std::move(builtin_method), self);
	if (!obj) { return Err(memory_error(sizeof(PyBuiltInMethod))); }
	return Ok(obj);
}

PyType *PyBuiltInMethod::type() const { return py::builtin_method(); }

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
	std::call_once(builtin_method_flag, []() { type = register_builtin_method(); });
	return std::move(type);
}

}// namespace py