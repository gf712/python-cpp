#include "PyProperty.hpp"
#include "AttributeError.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

PyObject *PyProperty::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == property())
	ASSERT(!kwargs || kwargs->map().empty())

	auto *fget = [&]() -> PyObject * {
		if (args) {
			ASSERT(args->size() == 1)
			return PyObject::from(args->elements()[0]);
		}
		return nullptr;
	}();

	return PyProperty::create(fget, nullptr, nullptr, nullptr);
}

PyProperty::PyProperty(PyObject *fget, PyObject *fset, PyObject *fdel, PyString *name)
	: PyBaseObject(BuiltinTypes::the().property()), m_getter(fget), m_setter(fset), m_deleter(fdel),
	  m_property_name(name)
{}

std::string PyProperty::to_string() const
{
	return fmt::format("<property object at {}>", static_cast<const void *>(this));
}

PyProperty *PyProperty::create(PyObject *fget, PyObject *fset, PyObject *fdel, PyString *name)
{
	return VirtualMachine::the().heap().allocate<PyProperty>(fget, fset, fdel, name);
}

PyObject *PyProperty::getter(PyTuple *args, PyDict *kwargs) const
{
	(void)args;
	(void)kwargs;
	TODO();
	return nullptr;
}

PyObject *PyProperty::setter(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->map().empty())
	ASSERT(args)
	ASSERT(args->size() == 1)

	auto *fgets = PyObject::from(args->elements()[0]);

	return PyProperty::create(fgets, nullptr, nullptr, m_property_name);
}

PyObject *PyProperty::deleter(PyTuple *args, PyDict *kwargs) const
{
	(void)args;
	(void)kwargs;
	TODO();
	return nullptr;
}

PyObject *PyProperty::__repr__() const { return PyString::create(PyProperty::to_string()); }

PyObject *PyProperty::__get__(PyObject *instance, PyObject *) const
{
	if (!instance || instance == py_none()) { return const_cast<PyProperty *>(this); }

	if (!m_getter) {
		if (m_property_name) {
			VirtualMachine::the().interpreter().raise_exception(
				attribute_error("unreadable attribute {}", m_property_name->to_string()));
		} else {
			VirtualMachine::the().interpreter().raise_exception(
				attribute_error("unreadable attribute"));
		}
		return nullptr;
	}

	return m_getter->call(PyTuple::create(instance), nullptr);
}

void PyProperty::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_getter) { visitor.visit(*m_getter); }
	if (m_setter) { visitor.visit(*m_setter); }
	if (m_deleter) { visitor.visit(*m_deleter); }
	if (m_property_name) { visitor.visit(*m_property_name); }
}

PyType *PyProperty::type() const { return property(); }

namespace {

std::once_flag property_flag;

std::unique_ptr<TypePrototype> register_property()
{
	return std::move(klass<PyProperty>("property")
						 .def("getter", &PyProperty::getter)
						 .def("deleter", &PyProperty::deleter)
						 .def("setter", &PyProperty::setter)
						 .type);
}
}// namespace

std::unique_ptr<TypePrototype> PyProperty::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(property_flag, []() { type = ::register_property(); });
	return std::move(type);
}

template<> PyProperty *py::as(PyObject *obj)
{
	if (obj->type() == property()) { return static_cast<PyProperty *>(obj); }
	return nullptr;
}

template<> const PyProperty *py::as(const PyObject *obj)
{
	if (obj->type() == property()) { return static_cast<const PyProperty *>(obj); }
	return nullptr;
}