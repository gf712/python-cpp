#include "PyStaticMethod.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "RuntimeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

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

PyResult PyStaticMethod::__repr__() const { return PyString::create(to_string()); }

PyResult PyStaticMethod::call_static_method(PyTuple *args, PyDict *kwargs)
{
	ASSERT(m_static_method->is_callable())
	return m_static_method->call(args, kwargs);
}

PyStaticMethod::PyStaticMethod(PyString *name, PyType *underlying_type, PyObject *function)
	: PyBaseObject(BuiltinTypes::the().static_method()), m_name(std::move(name)),
	  m_underlying_type(underlying_type), m_static_method(function)
{}


PyResult PyStaticMethod::create(PyString *name, PyObject *function)
{
	auto result = VirtualMachine::the().heap().allocate<PyStaticMethod>(name, nullptr, function);
	if (!result) { return PyResult::Err(memory_error(sizeof(PyStaticMethod))); }
	return PyResult::Ok(result);
}

PyResult PyStaticMethod::__get__(PyObject * /*instance*/, PyObject * /*owner*/) const
{
	// this check is probably not needed, but it is still here because CPython can raise this error
	if (!m_static_method) {
		return PyResult::Err(runtime_error("uninitialized staticmethod object"));
	}
	return PyResult::Ok(m_static_method);
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

template<> PyStaticMethod *py::as(PyObject *obj)
{
	if (obj->type() == static_method()) { return static_cast<PyStaticMethod *>(obj); }
	return nullptr;
}

template<> const PyStaticMethod *py::as(const PyObject *obj)
{
	if (obj->type() == static_method()) { return static_cast<const PyStaticMethod *>(obj); }
	return nullptr;
}