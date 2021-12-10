#include "PyType.hpp"
#include "CustomPyObject.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyList.hpp"
#include "PyMethodWrapper.hpp"
#include "PySlotWrapper.hpp"
#include "PyStaticMethod.hpp"
#include "PyString.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

template<> PyType *as(PyObject *obj)
{
	if (obj->type() == type()) { return static_cast<PyType *>(obj); }
	return nullptr;
}

template<> const PyType *as(const PyObject *obj)
{
	if (obj->type() == type()) { return static_cast<const PyType *>(obj); }
	return nullptr;
}

namespace {

TypePrototype clone(std::string name, const TypePrototype &prototype)
{
	TypePrototype type_prototype;
	type_prototype.__name__ = std::move(name);
#define COPY_SLOT(NAME) type_prototype.NAME = prototype.NAME;
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

std::unique_ptr<TypePrototype> register_type()
{
	return std::move(klass<PyType>("type").def("mro", &PyType::mro).type);
}

std::vector<PyObject *> merge(const std::vector<std::vector<PyObject *>> &mros)
{
	if (std::all_of(mros.begin(), mros.end(), [](const auto &vec) { return vec.empty(); })) {
		return {};
	}
	for (const auto &el : mros) {
		auto *candidate = el[0];
		auto candidate_not_in_mro_tail = [&candidate](const std::vector<PyObject *> &m) {
			if (m.size() > 1) {
				auto it = std::find_if(m.begin() + 1, m.end(), [&candidate](const PyObject *c) {
					return c == candidate;
				});
				return it == m.end();
			} else {
				return true;
			}
		};
		if (std::all_of(mros.begin(), mros.end(), candidate_not_in_mro_tail)) {
			std::vector<PyObject *> result;
			result.push_back(candidate);
			std::vector<std::vector<PyObject *>> rest;
			for (const auto &m : mros) {
				auto *head = m[0];
				if (head == candidate) {
					rest.push_back(std::vector<PyObject *>{ m.begin() + 1, m.end() });
				} else {
					rest.push_back(m);
				}
			}
			auto tmp = merge(rest);
			result.insert(result.end(), tmp.begin(), tmp.end());
			return result;
		}
	}

	// error
	TODO()
}

std::vector<PyObject *> mro_(PyType *type)
{
	if (type == custom_object()) { return { type }; }

	std::vector<PyObject *> mro_types;
	mro_types.push_back(type);

	std::vector<std::vector<PyObject *>> bases_mro;

	for (const auto &base : type->underlying_type().__bases__->elements()) {
		if (auto *precomputed_mro =
				static_cast<PyType *>(std::get<PyObject *>(base))->underlying_type().__mro__) {
			std::vector<PyObject *> base_mro;
			base_mro.reserve(precomputed_mro->size());
			for (const auto &el : precomputed_mro->elements()) {
				base_mro.push_back(std::get<PyObject *>(el));
			}
			bases_mro.push_back(base_mro);
		} else {
			bases_mro.push_back(mro_(as<PyType>(std::get<PyObject *>(base))));
		}
	}

	auto result = merge(bases_mro);
	mro_types.insert(mro_types.end(), result.begin(), result.end());

	return mro_types;
}
}// namespace


PyType::PyType(TypePrototype type_prototype)
	: PyBaseObject(BuiltinTypes::the().type()), m_underlying_type(type_prototype)
{}

PyType *PyType::type() const
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
	m_underlying_type.__class__ = this;
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

	if (!m_underlying_type.__bases__) {
		// not ideal, but avoids recursively calling custom_object()
		if (m_underlying_type.__name__ == "object") {
			m_underlying_type.__bases__ = PyTuple::create();
		} else {
			m_underlying_type.__bases__ = PyTuple::create(custom_object());
		}
	}
	m_attributes["__bases__"] = m_underlying_type.__bases__;
	m_underlying_type.__dict__->insert(PyString::create("__bases__"), m_underlying_type.__bases__);
	// not ideal, but avoids recursively calling custom_object()
	if (m_underlying_type.__name__ == "object") {
		m_underlying_type.__mro__ = PyTuple::create(this);
	} else {
		m_underlying_type.__mro__ = mro_internal();
	}
	m_attributes["__mro__"] = m_underlying_type.__mro__;
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

	if (!bases->elements().empty()) {
		std::unordered_set<PyObject *> bases_set;
		for (const auto &b : bases->elements()) {
			ASSERT(std::holds_alternative<PyObject *>(b))
			if (bases_set.contains(std::get<PyObject *>(b))) {
				auto *duplicate_type = as<PyType>(std::get<PyObject *>(b));
				VirtualMachine::the().interpreter().raise_exception(type_error(
					"duplicate base class {}", duplicate_type->underlying_type().__name__));
				return nullptr;
			}
			bases_set.insert(std::get<PyObject *>(b));
		}
	}

	auto *type = VirtualMachine::the().heap().allocate<PyType>(
		clone(name->value(), custom_object()->underlying_type()));

	if (bases->elements().empty()) {
		// all objects inherit from object by default
		bases = PyTuple::create(custom_object());
	}
	type->m_underlying_type.__bases__ = bases;
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
	// if (bases->size() > 0) { TODO() }

	return type;
}

PyObject *PyType::__call__(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->map().size() == 0)
	if (this == ::type()) {
		if (args->size() == 1) { return PyObject::from(args->elements()[0])->type(); }
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

	// If __new__() does not return an instance of cls, then the new instance’s __init__() method
	// will not be invoked.
	if (obj->type() == this) {
		// If __new__() is invoked during object construction and it returns an instance of cls,
		// then the new instance’s __init__() method will be invoked like __init__(self[, ...]),
		// where self is the new instance and the remaining arguments are the same as were passed to
		// the object constructor.
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

PyTuple *PyType::mro_internal()
{
	if (!m_underlying_type.__mro__) {
		const auto &result = mro_(this);
		m_underlying_type.__mro__ = PyTuple::create(result);
	}
	return m_underlying_type.__mro__;
}

PyList *PyType::mro() { return PyList::create(mro_internal()->elements()); }

bool PyType::issubclass(const PyType *other)
{
	if (this == other) { return true; }

	// every type is a subclass of object
	if (other == custom_object()) { return true; }

	auto *this_mro = mro_internal();
	for (const auto &el : this_mro->elements()) {
		if (std::get<PyObject *>(el) == other) { return true; }
	}

	return false;
}
