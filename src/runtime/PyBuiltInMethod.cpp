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

PyBuiltInMethod::PyBuiltInMethod(PyType *type) : PyBaseObject(type) {}

PyBuiltInMethod::PyBuiltInMethod(MethodDefinition &method_definition, PyObject *self)
	: PyBaseObject(BuiltinTypes::the().builtin_method()), m_ml(method_definition), m_self(self)
{}

void PyBuiltInMethod::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_self);
}

std::string PyBuiltInMethod::to_string() const
{
	ASSERT(m_ml);

	return fmt::format("<built-in method '{}' of '{}' object at {}>",
		m_ml->get().name,
		m_self->type()->name(),
		static_cast<const void *>(this));
}

PyResult<PyObject *> PyBuiltInMethod::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyBuiltInMethod::__call__(PyTuple *args, PyDict *kwargs)
{
	ASSERT(m_ml);

	if (m_ml->get().flags.flags() == MethodFlags::create().flags()) {
		return m_ml->get().method(m_self, args, kwargs);
	} else if (m_ml->get().flags.is_set(MethodFlags::Flag::CLASSMETHOD)) {
		return m_ml->get().method(m_self->type(), args, kwargs);
	} else {
		TODO();
	}
}

PyResult<PyBuiltInMethod *> PyBuiltInMethod::create(MethodDefinition &method_definition,
	PyObject *self)
{
	auto *obj = VirtualMachine::the().heap().allocate<PyBuiltInMethod>(method_definition, self);
	if (!obj) { return Err(memory_error(sizeof(PyBuiltInMethod))); }
	return Ok(obj);
}

PyType *PyBuiltInMethod::static_type() const { return py::builtin_method(); }

namespace {

	std::once_flag builtin_method_flag;

	std::unique_ptr<TypePrototype> register_builtin_method()
	{
		return std::move(klass<PyBuiltInMethod>("builtin_method").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyBuiltInMethod::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(builtin_method_flag, []() { type = register_builtin_method(); });
		return std::move(type);
	};
}

}// namespace py
