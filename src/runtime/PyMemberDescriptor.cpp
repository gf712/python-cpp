#include "PyMemberDescriptor.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyMemberDescriptor *as(PyObject *obj)
{
	if (obj->type() == member_descriptor()) { return static_cast<PyMemberDescriptor *>(obj); }
	return nullptr;
}

template<> const PyMemberDescriptor *as(const PyObject *obj)
{
	if (obj->type() == member_descriptor()) { return static_cast<const PyMemberDescriptor *>(obj); }
	return nullptr;
}

PyMemberDescriptor::PyMemberDescriptor(PyType *type) : PyBaseObject(type) {}

PyMemberDescriptor::PyMemberDescriptor(PyString *name,
	PyType *underlying_type,
	std::function<PyResult<PyObject *>(PyObject *)> member,
	std::function<PyResult<std::monostate>(PyObject *, PyObject *)> setter)
	: PyBaseObject(BuiltinTypes::the().member_descriptor()), m_name(std::move(name)),
	  m_underlying_type(underlying_type), m_member_accessor(std::move(member)),
	  m_member_setter(std::move(setter))
{}

PyResult<PyMemberDescriptor *> PyMemberDescriptor::create(PyString *name,
	PyType *underlying_type,
	std::function<PyResult<PyObject *>(PyObject *)> member,
	std::function<PyResult<std::monostate>(PyObject *, PyObject *)> setter)
{
	auto *obj = VirtualMachine::the().heap().allocate<PyMemberDescriptor>(
		name, underlying_type, member, setter);
	if (!obj) { return Err(memory_error(sizeof(PyMemberDescriptor))); }
	return Ok(obj);
}

void PyMemberDescriptor::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	visitor.visit(*m_underlying_type);
}

std::string PyMemberDescriptor::to_string() const
{
	return fmt::format(
		"<member '{}' of '{}' objects>", m_name->to_string(), m_underlying_type->name());
}

PyResult<PyObject *> PyMemberDescriptor::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyMemberDescriptor::__get__(PyObject *instance, PyObject * /*owner*/) const
{
	if (!instance) { return Ok(const_cast<PyMemberDescriptor *>(this)); }
	if ((instance->type() != m_underlying_type)
		&& !instance->type()->issubclass(m_underlying_type)) {
		return Err(
			type_error("descriptor '{}' for '{}' objects "
					   "doesn't apply to a '{}' object",
				m_name->value(),
				m_underlying_type->underlying_type().__name__,
				instance->type()->underlying_type().__name__));
	}

	return m_member_accessor(instance);
}

PyResult<std::monostate> PyMemberDescriptor::__set__(PyObject *obj, PyObject *value)
{
	if (obj->type() != m_underlying_type && !obj->type()->issubclass(m_underlying_type)) {
		return Err(type_error("descriptor '{}' for '{}' objects doesn't apply to a '{}' object",
			m_name->value(),
			m_underlying_type->underlying_type().__name__,
			obj->type()->underlying_type().__name__));
	}

	return m_member_setter(obj, value);
}


PyType *PyMemberDescriptor::static_type() const { return member_descriptor(); }

namespace {

	std::once_flag method_wrapper_flag;

	std::unique_ptr<TypePrototype> register_member_descriptor()
	{
		return std::move(klass<PyMemberDescriptor>("member_descriptor").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyMemberDescriptor::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(method_wrapper_flag, []() { type = register_member_descriptor(); });
		return std::move(type);
	};
}

}// namespace py
