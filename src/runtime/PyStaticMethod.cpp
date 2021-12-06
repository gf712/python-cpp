#include "PyStaticMethod.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

std::string PyStaticMethod::to_string() const
{
	return fmt::format(
		"<staticmethod '{}' of '{}' objects>", m_name->to_string(), m_underlying_type->name());
}

void PyStaticMethod::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	visitor.visit(*m_underlying_type);
}

PyObject *PyStaticMethod::__repr__() const { return PyString::create(to_string()); }

PyObject *PyStaticMethod::__call__(PyTuple *args, PyDict *kwargs)
{
	// split args tuple -> (args[0], args[1:])
	// since args[0] is type (right?)
	// std::vector<Value> new_args_vector;
	// new_args_vector.reserve(args->size() - 1);
	// PyObject *type = PyObject::from(args->elements()[0]);
	// for (size_t i = 1; i < args->size(); ++i) { new_args_vector.push_back(args->elements()[i]); }
	// args = PyTuple::create(new_args_vector);

	auto *result = m_static_method(m_underlying_type, args, kwargs);

	// FIXME: this should be independent from the VM's registers
	VirtualMachine::the().reg(0) = result;

	return result;
}

PyStaticMethod::PyStaticMethod(PyString *name,
	PyType *underlying_type,
	std::function<PyObject *(PyType *, PyTuple *, PyDict *)> function)
	: PyBaseObject(PyObjectType::PY_SLOT_WRAPPER, BuiltinTypes::the().slot_wrapper()),
	  m_name(std::move(name)), m_underlying_type(underlying_type), m_static_method(std::move(function))
{}

PyStaticMethod *PyStaticMethod::create(PyString *name,
	PyType *underlying_type,
	std::function<PyObject *(PyType *, PyTuple *, PyDict *)> function)
{
	return VirtualMachine::the().heap().allocate<PyStaticMethod>(name, underlying_type, function);
}

PyType *PyStaticMethod::type_() const { return ::static_method(); }

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