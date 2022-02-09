#include "PyClassMethod.hpp"
#include "PyDict.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "RuntimeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;


PyObject *PyClassMethod::__new__(const PyType *type, PyTuple *, PyDict *)
{
	ASSERT(type == classmethod())

	return PyClassMethod::create();
}

std::optional<int32_t> PyClassMethod::__init__(PyTuple *args, PyDict *kwargs)
{
	ASSERT(args && args->size() == 1)
	ASSERT(!kwargs || kwargs->map().empty())

	auto *callable = PyObject::from(args->elements()[0]);

	m_callable = callable;

	return 0;
}

PyClassMethod::PyClassMethod() : PyBaseObject(BuiltinTypes::the().classmethod()) {}

std::string PyClassMethod::to_string() const
{
	return fmt::format("<classmethod object at {}>", static_cast<const void *>(this));
}

PyClassMethod *PyClassMethod::create()
{
	return VirtualMachine::the().heap().allocate<PyClassMethod>();
}

PyObject *PyClassMethod::__repr__() const { return PyString::create(PyClassMethod::to_string()); }

PyObject *PyClassMethod::__get__(PyObject *instance, PyObject *owner) const
{
	if (!m_callable) {
		VirtualMachine::the().interpreter().raise_exception(
			runtime_error("uninitalized classmethod object"));
		return nullptr;
	}

	if (!owner) { owner = instance->type(); }

	if (m_callable->type_prototype().__get__.has_value()) { return m_callable->get(owner, owner); }

	TODO();
	return nullptr;
}

void PyClassMethod::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_callable) visitor.visit(*m_callable);
}

PyType *PyClassMethod::type() const { return classmethod(); }

namespace {

std::once_flag classmethod_flag;

std::unique_ptr<TypePrototype> register_classmethod()
{
	return std::move(klass<PyClassMethod>("classmethod").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyClassMethod::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(classmethod_flag, []() { type = ::register_classmethod(); });
	return std::move(type);
}

template<> PyClassMethod *py::as(PyObject *obj)
{
	if (obj->type() == classmethod()) { return static_cast<PyClassMethod *>(obj); }
	return nullptr;
}

template<> const PyClassMethod *py::as(const PyObject *obj)
{
	if (obj->type() == classmethod()) { return static_cast<const PyClassMethod *>(obj); }
	return nullptr;
}