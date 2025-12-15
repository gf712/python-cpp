#include "PyClassMethodDescriptor.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> PyClassMethodDescriptor *as(PyObject *obj)
{
	if (obj->type() == types::classmethod_descriptor()) {
		return static_cast<PyClassMethodDescriptor *>(obj);
	}
	return nullptr;
}

template<> const PyClassMethodDescriptor *as(const PyObject *obj)
{
	if (obj->type() == types::classmethod_descriptor()) {
		return static_cast<const PyClassMethodDescriptor *>(obj);
	}
	return nullptr;
}

PyClassMethodDescriptor::PyClassMethodDescriptor(PyType *type) : PyBaseObject(type) {}

PyClassMethodDescriptor::PyClassMethodDescriptor(PyString *name,
	PyType *underlying_type,
	MethodDefinition &method_definition,
	std::vector<PyObject *> &&captures)
	: PyBaseObject(types::BuiltinTypes::the().classmethod_descriptor()), m_name(std::move(name)),
	  m_underlying_type(underlying_type), m_method(method_definition),
	  m_captures(std::move(captures))
{}


void PyClassMethodDescriptor::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	visitor.visit(*m_underlying_type);
	for (auto *capture : m_captures) { visitor.visit(*capture); }
}

std::string PyClassMethodDescriptor::to_string() const
{
	return fmt::format(
		"<method '{}' of '{}' objects>", m_name->to_string(), m_underlying_type->name());
}

PyResult<PyObject *> PyClassMethodDescriptor::__repr__() const
{
	return PyString::create(to_string());
}

PyResult<PyObject *> PyClassMethodDescriptor::__call__(PyTuple *args, PyDict *kwargs)
{
	// split args tuple -> (args[0], args[1:])
	// since args[0] is cls (hopefully)
	std::vector<Value> new_args_vector;
	new_args_vector.reserve(args->size() - 1);
	auto cls_ = PyObject::from(args->elements()[0]);
	if (cls_.is_err()) return cls_;
	auto *cls = cls_.unwrap();
	for (size_t i = 1; i < args->size(); ++i) { new_args_vector.push_back(args->elements()[i]); }
	auto args_ = PyTuple::create(new_args_vector);
	if (args_.is_err()) return args_;
	args = args_.unwrap();

	ASSERT(m_method);
	return m_method->get().method(cls, args, kwargs);
}

PyResult<PyObject *> PyClassMethodDescriptor::__get__(PyObject *object, PyObject *type) const
{
	if (!type) {
		if (object) {
			type = object->type();
		} else {
			return Err(type_error("descriptor '{}' for type {} needs either an object or a type",
				m_name->value(),
				m_underlying_type->name()));
		}
	}
	if (!as<PyType>(type)) {
		return Err(type_error("descriptor '{}' for type '{}' needs a type, not a {} as arg 2",
			m_name->value(),
			m_underlying_type->name(),
			type->type()->name()));
	}
	if (!as<PyType>(type)->issubclass(m_underlying_type)) {
		return Err(type_error("descriptor '{}' requires a subtype of '{}' but received '{}'",
			m_name->value(),
			m_underlying_type->name(),
			type->type()->name()));
	}
	auto *cls = m_underlying_type;

	return PyNativeFunction::create(
		m_name->value(),
		[this, cls](PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
			std::vector<Value> new_args_vector;
			new_args_vector.reserve(args->size() + 1);
			new_args_vector.push_back(cls);
			new_args_vector.insert(
				new_args_vector.end(), args->elements().begin(), args->elements().end());
			auto args_ = PyTuple::create(new_args_vector);
			if (args_.is_err()) return args_;
			args = args_.unwrap();
			return const_cast<PyClassMethodDescriptor *>(this)->__call__(args, kwargs);
		},
		const_cast<PyClassMethodDescriptor *>(this),
		cls);
}


PyType *PyClassMethodDescriptor::static_type() const { return types::classmethod_descriptor(); }

namespace {

	std::once_flag classmethod_wrapper_flag;

	std::unique_ptr<TypePrototype> register_classmethod_wrapper()
	{
		return std::move(klass<PyClassMethodDescriptor>("classmethod_descriptor").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyClassMethodDescriptor::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(classmethod_wrapper_flag, []() { type = register_classmethod_wrapper(); });
		return std::move(type);
	};
}

}// namespace py
