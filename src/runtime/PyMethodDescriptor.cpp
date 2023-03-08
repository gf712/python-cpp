#include "PyMethodDescriptor.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> PyMethodDescriptor *as(PyObject *obj)
{
	if (obj->type() == method_wrapper()) { return static_cast<PyMethodDescriptor *>(obj); }
	return nullptr;
}

template<> const PyMethodDescriptor *as(const PyObject *obj)
{
	if (obj->type() == method_wrapper()) { return static_cast<const PyMethodDescriptor *>(obj); }
	return nullptr;
}

PyMethodDescriptor::PyMethodDescriptor(PyType *type) : PyBaseObject(type) {}

PyMethodDescriptor::PyMethodDescriptor(PyString *name,
	PyType *underlying_type,
	MethodDefinition &method_definition,
	std::vector<PyObject *> &&captures)
	: PyBaseObject(BuiltinTypes::the().method_wrapper()), m_name(std::move(name)),
	  m_underlying_type(underlying_type), m_method(method_definition),
	  m_captures(std::move(captures))
{}

PyResult<PyMethodDescriptor *> PyMethodDescriptor::create(PyString *name,
	PyType *underlying_type,
	MethodDefinition &method,
	std::vector<PyObject *> &&captures)
{
	auto *obj = VirtualMachine::the().heap().allocate<PyMethodDescriptor>(
		name, underlying_type, method, std::move(captures));
	if (!obj) { return Err(memory_error(sizeof(PyMethodDescriptor))); }
	return Ok(obj);
}

void PyMethodDescriptor::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_name) visitor.visit(*m_name);
	if (m_underlying_type) visitor.visit(*m_underlying_type);
	for (auto *capture : m_captures) { visitor.visit(*capture); }
}

std::string PyMethodDescriptor::to_string() const
{
	return fmt::format(
		"<method '{}' of '{}' objects>", m_name->to_string(), m_underlying_type->name());
}

PyResult<PyObject *> PyMethodDescriptor::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyMethodDescriptor::__call__(PyTuple *args, PyDict *kwargs)
{
	// split args tuple -> (args[0], args[1:])
	// since args[0] is self (hopefully)
	std::vector<Value> new_args_vector;
	new_args_vector.reserve(args->size() - 1);
	auto self_ = PyObject::from(args->elements()[0]);
	if (self_.is_err()) return self_;
	auto *self = self_.unwrap();
	for (size_t i = 1; i < args->size(); ++i) { new_args_vector.push_back(args->elements()[i]); }
	auto args_ = PyTuple::create(new_args_vector);
	if (args_.is_err()) return args_;
	args = args_.unwrap();

	ASSERT(m_method);
	return m_method->get().method(self, args, kwargs);
}

PyResult<PyObject *> PyMethodDescriptor::__get__(PyObject *instance, PyObject * /*owner*/) const
{
	if (!instance) { return Ok(const_cast<PyMethodDescriptor *>(this)); }
	if ((instance->type() != m_underlying_type)
		&& !instance->type()->issubclass(m_underlying_type)) {
		return Err(
			type_error("descriptor '{}' for '{}' objects "
					   "doesn't apply to a '{}' object",
				m_name->value(),
				m_underlying_type->underlying_type().__name__,
				instance->type()->underlying_type().__name__));
	}
	return PyNativeFunction::create(
		m_name->value(),
		[this, instance](PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
			std::vector<Value> new_args_vector;
			new_args_vector.reserve((args ? args->size() : 0) + 1);
			new_args_vector.push_back(instance);
			if (args) {
				new_args_vector.insert(
					new_args_vector.end(), args->elements().begin(), args->elements().end());
			}
			auto args_ = PyTuple::create(new_args_vector);
			if (args_.is_err()) return args_;
			args = args_.unwrap();
			return const_cast<PyMethodDescriptor *>(this)->__call__(args, kwargs);
		},
		const_cast<PyMethodDescriptor *>(this),
		instance);
}


PyType *PyMethodDescriptor::static_type() const { return method_wrapper(); }

namespace {

	std::once_flag method_wrapper_flag;

	std::unique_ptr<TypePrototype> register_method_wrapper()
	{
		return std::move(klass<PyMethodDescriptor>("method_descriptor").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyMethodDescriptor::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(method_wrapper_flag, []() { type = register_method_wrapper(); });
		return std::move(type);
	};
}

}// namespace py
