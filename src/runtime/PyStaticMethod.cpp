#include "PyStaticMethod.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

std::string PyStaticMethod::to_string() const
{
	if (m_underlying_type) {
		return fmt::format(
			"<staticmethod '{}' of '{}' objects>", m_name->to_string(), m_underlying_type->name());
	} else {
		return fmt::format("<staticmethod at {}>", static_cast<const void *>(this));
	}
}

void PyStaticMethod::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	if (m_underlying_type) { visitor.visit(*m_underlying_type); }
	if (m_static_method) { visitor.visit(*m_static_method); }
}

PyObject *PyStaticMethod::__repr__() const { return PyString::create(to_string()); }

PyObject *PyStaticMethod::call_static_method(PyTuple *args, PyDict *kwargs)
{
	ASSERT(m_static_method->is_callable())
	return m_static_method->call(args, kwargs);
}

PyStaticMethod::PyStaticMethod(PyString *name, PyType *underlying_type, PyObject *function)
	: PyBaseObject(BuiltinTypes::the().static_method()), m_name(std::move(name)),
	  m_underlying_type(underlying_type), m_static_method(function)
{}


PyStaticMethod *PyStaticMethod::create(PyString *name, PyObject *function)
{
	return VirtualMachine::the().heap().allocate<PyStaticMethod>(name, nullptr, function);
}

PyObject *PyStaticMethod::__get__(PyObject * /*instance*/, PyObject * /*owner*/) const
{
	// this check is probably not needed, but it is still here because CPython can raise this error
	if (!m_static_method) {
		// TODO: should raise runtime error
		//       RuntimeError("uninitialized staticmethod object")
		return nullptr;
	}
	return m_static_method;
}

PyType *PyStaticMethod::type() const { return ::static_method(); }

namespace {

std::once_flag static_method_flag;

std::unique_ptr<TypePrototype> register_static_method()
{
	return std::move(klass<PyStaticMethod>("static_method").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyStaticMethod::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(static_method_flag, []() { type = ::register_static_method(); });
	return std::move(type);
}

template<> PyStaticMethod *as(PyObject *obj)
{
	if (obj->type() == static_method()) { return static_cast<PyStaticMethod *>(obj); }
	return nullptr;
}

template<> const PyStaticMethod *as(const PyObject *obj)
{
	if (obj->type() == static_method()) { return static_cast<const PyStaticMethod *>(obj); }
	return nullptr;
}