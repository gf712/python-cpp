#include "PyGetSetDescriptor.hpp"
#include "AttributeError.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyGetSetDescriptor *as(PyObject *obj)
{
	if (obj->type() == getset_descriptor()) { return static_cast<PyGetSetDescriptor *>(obj); }
	return nullptr;
}

template<> const PyGetSetDescriptor *as(const PyObject *obj)
{
	if (obj->type() == getset_descriptor()) { return static_cast<const PyGetSetDescriptor *>(obj); }
	return nullptr;
}

PyGetSetDescriptor::PyGetSetDescriptor(PyString *name,
	PyType *underlying_type,
	PropertyDefinition &getset)
	: PyBaseObject(BuiltinTypes::the().getset_descriptor()), m_name(std::move(name)),
	  m_underlying_type(underlying_type), m_getset(getset)
{}

PyResult<PyGetSetDescriptor *>
	PyGetSetDescriptor::create(PyString *name, PyType *underlying_type, PropertyDefinition &getset)
{
	auto *obj =
		VirtualMachine::the().heap().allocate<PyGetSetDescriptor>(name, underlying_type, getset);
	if (!obj) { return Err(memory_error(sizeof(PyGetSetDescriptor))); }
	return Ok(obj);
}

void PyGetSetDescriptor::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	visitor.visit(*m_underlying_type);
}

std::string PyGetSetDescriptor::to_string() const
{
	return fmt::format(
		"<attribute '{}' of '{}' objects>", m_name->to_string(), m_underlying_type->name());
}

PyResult<PyObject *> PyGetSetDescriptor::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyGetSetDescriptor::__get__(PyObject *instance, PyObject * /*owner*/) const
{
	if (!instance) { return Ok(const_cast<PyGetSetDescriptor *>(this)); }
	if ((instance->type() != m_underlying_type)
		&& !instance->type()->issubclass(m_underlying_type)) {
		return Err(
			type_error("descriptor '{}' for '{}' objects "
					   "doesn't apply to a '{}' object",
				m_name->value(),
				m_underlying_type->underlying_type().__name__,
				instance->type()->underlying_type().__name__));
	}

	if (m_getset.member_getter.has_value()) {
		return Ok(m_getset.member_getter->operator()(instance));
	}

	return Err(attribute_error("attribute '{}' of '{}' objects is not readable",
		m_name->value(),
		m_underlying_type->name()));
}

PyResult<std::monostate> PyGetSetDescriptor::__set__(PyObject *obj, PyObject *value)
{
	if (obj->type() != m_underlying_type && !obj->type()->issubclass(m_underlying_type)) {
		return Err(type_error("descriptor '{}' for '{}' objects doesn't apply to a '{}' object",
			m_name->value(),
			m_underlying_type->underlying_type().__name__,
			obj->type()->underlying_type().__name__));
	}

	if (m_getset.member_setter.has_value()) {
		return m_getset.member_setter->operator()(obj, value);
	}

	return Err(attribute_error("attribute '{}' of '{}' objects is not writable",
		m_name->value(),
		m_underlying_type->name()));
}


PyType *PyGetSetDescriptor::type() const { return getset_descriptor(); }

namespace {

	std::once_flag getset_descriptor_flag;

	std::unique_ptr<TypePrototype> register_getset_descriptor()
	{
		return std::move(klass<PyGetSetDescriptor>("getset_descriptor").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyGetSetDescriptor::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(getset_descriptor_flag, []() { type = register_getset_descriptor(); });
		return std::move(type);
	};
}

}// namespace py