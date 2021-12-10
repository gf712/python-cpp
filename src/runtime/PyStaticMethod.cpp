#include "PyStaticMethod.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
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
}

PyObject *PyStaticMethod::__repr__() const { return PyString::create(to_string()); }

PyObject *PyStaticMethod::call_static_method(PyTuple *args, PyDict *kwargs)
{
	auto *result = [this](PyTuple *args, PyDict *kwargs) -> PyObject * {
		if (std::holds_alternative<TypeBoundFunctionType>(m_static_method)) {
			ASSERT(m_underlying_type)
			return std::get<TypeBoundFunctionType>(m_static_method)(
				m_underlying_type, args, kwargs);
		} else {
			ASSERT(!m_underlying_type)
			return std::get<FreeFunctionType>(m_static_method)(args, kwargs);
		}
	}(args, kwargs);

	// FIXME: this should be independent from the VM's registers
	VirtualMachine::the().reg(0) = result;

	return result;
}

PyStaticMethod::PyStaticMethod(PyString *name,
	PyType *underlying_type,
	std::variant<TypeBoundFunctionType, FreeFunctionType> function)
	: PyBaseObject(BuiltinTypes::the().slot_wrapper()), m_name(std::move(name)),
	  m_underlying_type(underlying_type), m_static_method(std::move(function))
{}

PyStaticMethod *
	PyStaticMethod::create(PyString *name, PyType *underlying_type, TypeBoundFunctionType function)
{
	return VirtualMachine::the().heap().allocate<PyStaticMethod>(name, underlying_type, function);
}

PyStaticMethod *PyStaticMethod::create(PyString *name, FreeFunctionType function)
{
	return VirtualMachine::the().heap().allocate<PyStaticMethod>(name, nullptr, function);
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