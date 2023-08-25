#include "PyStaticMethod.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "RuntimeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

PyStaticMethod::PyStaticMethod(PyType *type) : PyBaseObject(type) {}

PyResult<PyObject *> PyStaticMethod::__new__(const PyType *, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty())
	ASSERT(args && args->elements().size() == 1);

	return PyObject::from(args->elements()[0]).and_then([](PyObject *function) {
		return PyStaticMethod::create(function);
	});
}

void PyStaticMethod::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_underlying_type) { visitor.visit(*m_underlying_type); }
	if (m_static_method) { visitor.visit(*m_static_method); }
}

PyResult<PyObject *> PyStaticMethod::call_static_method(PyTuple *args, PyDict *kwargs)
{
	ASSERT(m_static_method->is_callable())
	return m_static_method->call(args, kwargs);
}

PyStaticMethod::PyStaticMethod(PyType *underlying_type, PyObject *function)
	: PyBaseObject(types::BuiltinTypes::the().static_method()), m_underlying_type(underlying_type),
	  m_static_method(function)
{}

PyResult<PyStaticMethod *> PyStaticMethod::create(PyObject *function)
{
	auto result = VirtualMachine::the().heap().allocate<PyStaticMethod>(nullptr, function);
	if (!result) { return Err(memory_error(sizeof(PyStaticMethod))); }
	return Ok(result);
}

PyResult<PyObject *> PyStaticMethod::__get__(PyObject * /*instance*/, PyObject * /*owner*/) const
{
	// this check is probably not needed, but it is still here because CPython can raise this error
	if (!m_static_method) { return Err(runtime_error("uninitialized staticmethod object")); }
	return Ok(m_static_method);
}

PyType *PyStaticMethod::static_type() const { return types::static_method(); }

namespace {

	std::once_flag static_method_flag;

	std::unique_ptr<TypePrototype> register_static_method()
	{
		return std::move(klass<PyStaticMethod>("staticmethod").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyStaticMethod::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(static_method_flag, []() { type = register_static_method(); });
		return std::move(type);
	};
}

template<> PyStaticMethod *as(PyObject *obj)
{
	if (obj->type() == types::static_method()) { return static_cast<PyStaticMethod *>(obj); }
	return nullptr;
}

template<> const PyStaticMethod *as(const PyObject *obj)
{
	if (obj->type() == types::static_method()) { return static_cast<const PyStaticMethod *>(obj); }
	return nullptr;
}
}// namespace py
