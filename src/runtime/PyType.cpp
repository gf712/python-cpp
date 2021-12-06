#include "PyType.hpp"
#include "CustomPyObject.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyMethodWrapper.hpp"
#include "PySlotWrapper.hpp"
#include "PyStaticMethod.hpp"
#include "PyString.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace {

TypePrototype clone(std::string name, const TypePrototype &prototype)
{
	TypePrototype type_prototype;
	type_prototype.__name__ = std::move(name);
#define COPY_SLOT(NAME) type_prototype. NAME = prototype. NAME;
	COPY_SLOT(__repr__)
	COPY_SLOT(__call__)
	COPY_SLOT(__new__)
	COPY_SLOT(__init__)
	COPY_SLOT(__hash__)
	COPY_SLOT(__lt__)
	COPY_SLOT(__le__)
	COPY_SLOT(__eq__)
	COPY_SLOT(__ne__)
	COPY_SLOT(__gt__)
	COPY_SLOT(__ge__)
	COPY_SLOT(__iter__)
	COPY_SLOT(__next__)
	COPY_SLOT(__len__)
	COPY_SLOT(__add__)
	COPY_SLOT(__sub__)
	COPY_SLOT(__mul__)
	COPY_SLOT(__exp__)
	COPY_SLOT(__lshift__)
	COPY_SLOT(__mod__)
	COPY_SLOT(__abs__)
	COPY_SLOT(__neg__)
	COPY_SLOT(__pos__)
	COPY_SLOT(__invert__)
	COPY_SLOT(__bool__)
#undef COPY_SLOT
	return type_prototype;
}

std::once_flag type_flag;

std::unique_ptr<TypePrototype> register_type() { return std::move(klass<PyType>("type").type); }
}// namespace


PyType::PyType(TypePrototype type_prototype)
	: PyBaseObject(PyObjectType::PY_TYPE, BuiltinTypes::the().type()),
	  m_underlying_type(type_prototype)
{}

PyType *PyType::type_() const
{
	// FIXME: probably not the best way to do this
	//		  this avoids infinite recursion where PyType representing "type" has type "type"
	if (m_underlying_type.__name__ == "type") {
		return const_cast<PyType *>(this);// :(
	} else {
		return ::type();
	}
}

PyType *PyType::initialize(TypePrototype type_prototype)
{
	auto *type = VirtualMachine::the().heap().allocate_static<PyType>(type_prototype).get();
	type->initialize();
	return type;
}

std::unique_ptr<TypePrototype> PyType::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(type_flag, []() { type = ::register_type(); });
	return std::move(type);
}

void PyType::initialize()
{
	// FIXME: this is only allocated in static memory so that the lifetime of __dict__
	//        matches the one of m_underlying_type
	m_underlying_type.__dict__ = VirtualMachine::the().heap().allocate_static<PyDict>().get();
	m_attributes["__dict__"] = m_underlying_type.__dict__;
	if (m_underlying_type.__add__.has_value()) {
		auto *name = PyString::create("__add__");
		auto add_func =
			PySlotWrapper::create(name, this, [this](PyObject *self, PyTuple *args, PyDict *) {
				ASSERT(args->size() == 1)
				return (*m_underlying_type.__add__)(self, PyObject::from(args->elements()[0]));
			});
		m_attributes["__add__"] = add_func;
		m_underlying_type.__dict__->insert(name, add_func);
	}
	if (m_underlying_type.__call__.has_value()) {
		auto *name = PyString::create("__call__");
		auto *call_fn = PySlotWrapper::create(
			name, this, [this](PyObject *self, PyTuple *args, PyDict *kwargs) {
				return (*m_underlying_type.__call__)(self, args, kwargs);
			});
		m_attributes["__call__"] = call_fn;
		m_underlying_type.__dict__->insert(name, call_fn);
	}
	if (m_underlying_type.__new__.has_value()) {
		auto *name = PyString::create("__new__");
		auto *call_fn =
			PyStaticMethod::create(name, this, [this](PyType *type, PyTuple *args, PyDict *kwargs) {
				return (*m_underlying_type.__new__)(type, args, kwargs);
			});
		m_attributes["__new__"] = call_fn;
		m_underlying_type.__dict__->insert(name, call_fn);
	}
	for (auto method : m_underlying_type.__methods__) {
		auto *name = PyString::create(method.name);
		auto *method_fn = PyMethodWrapper::create(name, this, method.method);
		m_underlying_type.__dict__->insert(name, method_fn);
		m_attributes[method.name] = method_fn;
	}
}

PyObject *PyType::new_(PyTuple *args, PyDict *kwargs) const
{
	if (auto it = m_underlying_type.__dict__->map().find(String{ "__new__" });
		it != m_underlying_type.__dict__->map().end()) {
		ASSERT(std::holds_alternative<PyObject *>(it->second))
		auto *obj = std::get<PyObject *>(it->second);
		// prepend class type to args tuple -> obj->call((cls, *args), kwargs)
		std::vector<Value> args_with_type;
		args_with_type.reserve(args->size() + 1);
		// FIXME: remove this const_cast. Either args are const or PyType::new_ should not be const
		args_with_type.push_back(const_cast<PyType *>(this));
		for (const auto &el : args->elements()) { args_with_type.push_back(el); }
		args = PyTuple::create(args_with_type);
		return obj->call(args, kwargs);
	} else if (m_underlying_type.__new__.has_value()) {
		return m_underlying_type.__new__->operator()(this, args, kwargs);
	}
	return nullptr;
}


PyObject *PyType::__new__(const PyType *type_, PyTuple *args, PyDict *kwargs)
{
	(void)type_;
	ASSERT(args && args->size() == 3)
	ASSERT(!kwargs || kwargs->map().empty())

	auto *name = as<PyString>(PyObject::from(args->elements()[0]));
	ASSERT(name)
	auto *bases = as<PyTuple>(PyObject::from(args->elements()[1]));
	ASSERT(bases)
	auto *ns = as<PyDict>(PyObject::from(args->elements()[2]));
	ASSERT(ns)

	if (bases->size() > 0) { TODO() }

	auto *type =
		VirtualMachine::the().heap().allocate<PyType>(clone(name->value(), custom_object()->underlying_type()));
	type->initialize();

	for (const auto &[key, v] : ns->map()) {
		auto *attr_method_name = [](const auto &k) {
			if (std::holds_alternative<String>(k)) {
				return PyString::create(std::get<String>(k).s);
			} else if (std::holds_alternative<PyObject *>(k)) {
				auto *obj = std::get<PyObject *>(k);
				ASSERT(as<PyString>(obj))
				return as<PyString>(obj);
			} else {
				TODO()
			}
		}(key);

		type->m_underlying_type.__dict__->insert(attr_method_name, v);
	}

	return type;
}

PyObject *PyType::__call__(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->map().size() == 0)
	if (this == ::type()) {
		if (args->size() == 1) { return PyObject::from(args->elements()[0])->type_(); }
		if (args->size() != 3) {
			VirtualMachine::the().interpreter().raise_exception(
				type_error("type() takes 1 or 3 arguments, got {}", args->size()));
			return nullptr;
		}
	}

	auto *obj = new_(args, kwargs);
	if (!obj) {
		VirtualMachine::the().interpreter().raise_exception(
			fmt::format("cannot create '{}' instances", m_type_prototype.__name__));
		return nullptr;
	}

	// FIXME: this should be checking if it is subtype rather than the same type
	if (obj->type_() == this) {
		if (const auto res = obj->init(args, kwargs); res.has_value()) {
			if (*res < 0) {
				// error
				return nullptr;
			}
		}
	}
	return obj;
}

std::string PyType::to_string() const
{
	return fmt::format("<class '{}'>", m_underlying_type.__name__);
}

PyObject *PyType::__repr__() const { return PyString::create(to_string()); }