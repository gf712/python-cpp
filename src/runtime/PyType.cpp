#include "PyType.hpp"
#include "PyBoundMethod.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyInteger.hpp"
#include "PyList.hpp"
#include "PyMethodDescriptor.hpp"
#include "PyNone.hpp"
#include "PySlotWrapper.hpp"
#include "PyStaticMethod.hpp"
#include "PyString.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

template<> PyType *py::as(PyObject *obj)
{
	if (obj->type() == type()) { return static_cast<PyType *>(obj); }
	return nullptr;
}

template<> const PyType *py::as(const PyObject *obj)
{
	if (obj->type() == type()) { return static_cast<const PyType *>(obj); }
	return nullptr;
}

// FIXME: copied from PyObject.cpp.
// Ideally this would live somewhere where all classes could see it, but this would require
// reorganising the project structure, since some of the functions used here, are not known
// when PyObject is declared in PyObject.hpp
template<typename SlotFunctionType, typename... Args>
static PyObject *call_slot(const std::variant<SlotFunctionType, PyObject *> &slot, Args &&... args_)
{
	if (std::holds_alternative<SlotFunctionType>(slot)) {
		auto result = std::get<SlotFunctionType>(slot)(std::forward<Args>(args_)...);
		using ResultType = std::remove_pointer_t<decltype(result)>;

		if constexpr (std::is_base_of_v<PyObject, ResultType>) {
			return result;
		} else if constexpr (std::is_same_v<std::optional<int>, ResultType>) {
			ASSERT(result.has_value())
			return PyInteger::create(static_cast<int64_t>(*result));
		} else if constexpr (std::is_same_v<size_t, ResultType>) {
			return PyInteger::create(result);
		} else {
			[]<bool flag = false>() { static_assert(flag, "unsupported return type"); }
			();
		}
	} else if (std::holds_alternative<PyObject *>(slot)) {
		// FIXME: this const_cast is needed since in Python land there is no concept of
		//		  PyObject constness (right?). But for the internal calls handled above
		//		  which are resolved in the C++ runtime, we want to enforce constness
		//		  so we end up with the awkward line below. But how could we do better?
		auto *args = PyTuple::create(
			const_cast<PyObject *>(static_cast<const PyObject *>(std::forward<Args>(args_)))...);
		PyDict *kwargs = nullptr;
		return std::get<PyObject *>(slot)->call(args, kwargs);
	} else {
		TODO();
	}
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
	TODO();
}

std::vector<PyObject *> mro_(PyType *type)
{
	if (type == object()) { return { type }; }

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
	type->initialize(nullptr);
	return type;
}

std::unique_ptr<TypePrototype> PyType::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(type_flag, []() { type = ::register_type(); });
	return std::move(type);
}

namespace {
bool descriptor_is_data(const PyObject *obj)
{
	// FIXME: temporary hack to get object.__new__ working, but requires __set__ to be implemented
	//        should be:
	//        obj->type()->underlying_type().__set__.has_value()
	return !as<PyStaticMethod>(obj) && !as<PySlotWrapper>(obj);
}
}// namespace

PyObject *PyType::__getattribute__(PyObject *attribute) const
{
	auto *name = as<PyString>(attribute);
	if (!name) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("attribute name must be string, not '{}'",
				attribute->type()->underlying_type().__name__));
		return nullptr;
	}

	auto *meta_attr = type()->lookup(name);
	if (meta_attr) {
		const auto &meta_get = meta_attr->type()->underlying_type().__get__;
		if (meta_get.has_value() && descriptor_is_data(meta_attr)) {
			auto *result = call_slot(*meta_get, meta_attr, const_cast<PyType *>(this), type());
			// FIXME: returning null is valid, but this is useful for debugging sometimes
			//        may be worth checking first if a exception has been thrown at this point
			ASSERT(result)
			return result;
		}
	}

	auto *attr = lookup(name);
	if (attr) {
		const auto &local_get = attr->type()->underlying_type().__get__;
		if (local_get.has_value()) {
			auto *result = call_slot(*local_get, attr, nullptr, const_cast<PyType *>(this));
			// FIXME: returning null is valid, but this is useful for debugging sometimes
			//        may be worth checking first if a exception has been thrown at this point
			ASSERT(result)
			return result;
		}
		return attr;
	}

	if (meta_attr) { return meta_attr; }

	VirtualMachine::the().interpreter().raise_exception(type_error(
		"type object '{}' has no attribute '{}'", m_underlying_type.__name__, name->value()));
	return nullptr;
}

PyObject *PyType::lookup(PyObject *name) const
{
	for (const auto &t_ : mro_internal()->elements()) {
		ASSERT(std::holds_alternative<PyObject *>(t_))
		auto *t = as<PyType>(std::get<PyObject *>(t_));
		ASSERT(t)
		ASSERT(t->m_underlying_type.__dict__)
		const auto &dict = t->m_underlying_type.__dict__->map();
		if (auto it = dict.find(name); it != dict.end()) { return PyObject::from(it->second); }
	}
	return nullptr;
}

bool PyType::update_if_special(const std::string &name, const Value &value)
{
	if (!name.starts_with("__")) { return false; }
	if (name == "__new__") {
		m_underlying_type.__new__ = PyObject::from(value);
		return true;
	}
	if (name == "__init__") {
		m_underlying_type.__init__ = PyObject::from(value);
		return true;
	}
	if (name == "__repr__") {
		m_underlying_type.__repr__ = PyObject::from(value);
		return true;
	}
	return false;
}

void PyType::update_methods_and_class_attributes(PyDict *ns)
{
	if (ns) {
		for (const auto &[key, v] : ns->map()) {
			auto *attr_method_name = [](const auto &k) {
				if (std::holds_alternative<String>(k)) {
					return PyString::create(std::get<String>(k).s);
				} else if (std::holds_alternative<PyObject *>(k)) {
					auto *obj = std::get<PyObject *>(k);
					ASSERT(as<PyString>(obj))
					return as<PyString>(obj);
				} else {
					TODO();
				}
			}(key);

			update_if_special(attr_method_name->value(), v);
			m_underlying_type.__dict__->insert(attr_method_name, v);
		}
	}
}

namespace {
template<typename SlotFunctionType, typename FunctorType>
std::pair<String, PyObject *> wrap_slot(PyType *type,
	std::string_view name_,
	PyDict *ns,
	std::variant<SlotFunctionType, PyObject *> &slot,
	FunctorType &&f)
{
	String name_str{ std::string(name_) };
	if (ns) {
		if (auto it = ns->map().find(name_str); it != ns->map().end()) {
			slot = PyObject::from(it->second);
			return { name_str, std::get<PyObject *>(slot) };
		}
	}

	if (std::holds_alternative<SlotFunctionType>(slot)) {
		// FIXME: should PyString have a std::string_view constructor and not worry about the
		// 		  lifetime of the string_view?
		auto *name = PyString::create(std::string(name_));
		// the lifetime of the type is extended by the slot wrapper
		auto *func = PySlotWrapper::create(name, type, f);
		// FIXME: String should handle string_view, no need create a std::string here
		return { name_str, func };
	} else {
		return { name_str, std::get<PyObject *>(slot) };
	}
}
}// namespace

void PyType::initialize(PyDict *ns)
{
	// FIXME: this a hack to extend the lifetime of all objects owned by types, which *can* be
	// static (i.e. builtin types) and not visited by the garbage collector
	[[maybe_unused]] auto scope_static_alloc =
		VirtualMachine::the().heap().scoped_static_allocation();
	m_underlying_type.__class__ = this;
	m_underlying_type.__dict__ = PyDict::create();
	// m_attributes should be a "mappingproxy" object, not dict for PyType
	m_attributes = m_underlying_type.__dict__;
	if (m_underlying_type.__add__.has_value()) {
		auto [name, add_func] = wrap_slot(this,
			"__add__",
			ns,
			*m_underlying_type.__add__,
			[this](PyObject *self, PyTuple *args, PyDict *kwargs) {
				ASSERT(args && args->size() == 1)
				ASSERT(!kwargs || kwargs->map().empty())
				return std::get<AddSlotFunctionType>(*m_underlying_type.__add__)(
					self, PyObject::from(args->elements()[0]));
			});
		m_underlying_type.__dict__->insert(name, add_func);
	}
	if (m_underlying_type.__repr__.has_value()) {
		auto [name, repr_func] = wrap_slot(this,
			"__repr__",
			ns,
			*m_underlying_type.__repr__,
			[this](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyObject * {
				if (args && args->size() > 0) {
					VirtualMachine::the().interpreter().raise_exception(
						type_error("expected 0 arguments, got {}", args->size()));
					return nullptr;
				}
				ASSERT(!kwargs || kwargs->map().empty())
				return std::get<ReprSlotFunctionType>(*m_underlying_type.__repr__)(self);
			});
		m_underlying_type.__dict__->insert(name, repr_func);
	}
	if (m_underlying_type.__call__.has_value()) {
		auto [name, call_func] = wrap_slot(this,
			"__call__",
			ns,
			*m_underlying_type.__call__,
			[this](PyObject *self, PyTuple *args, PyDict *kwargs) {
				return std::get<CallSlotFunctionType>(*m_underlying_type.__call__)(
					self, args, kwargs);
			});
		m_underlying_type.__dict__->insert(name, call_func);
	}
	if (m_underlying_type.__init__.has_value()) {
		auto [name, init_func] = wrap_slot(this,
			"__init__",
			ns,
			*m_underlying_type.__init__,
			[this](PyObject *self, PyTuple *args, PyDict *kwargs) {
				auto result =
					std::get<InitSlotFunctionType>(*m_underlying_type.__init__)(self, args, kwargs);
				if (result) { ASSERT(*result == 0) }
				return py_none();
			});
		m_underlying_type.__dict__->insert(name, init_func);
	}
	if (m_underlying_type.__new__.has_value()) {
		auto *name = PyString::create("__new__");
		if (std::holds_alternative<NewSlotFunctionType>(*m_underlying_type.__new__)) {
			auto *fn = PyNativeFunction::create(
				"__new__",
				[this](PyTuple *args, PyDict *kwargs) -> PyObject * {
					auto *maybe_type = PyObject::from(args->elements()[0]);
					auto *type = as<PyType>(maybe_type);
					ASSERT(type)
					// pop out type from args
					std::vector<Value> new_args;
					new_args.reserve(args->size() - 1);
					new_args.insert(
						new_args.end(), args->elements().begin() + 1, args->elements().end());
					args = PyTuple::create(new_args);
					return std::get<NewSlotFunctionType>(*m_underlying_type.__new__)(
						type, args, kwargs);
				},
				this);
			auto *new_fn = PyStaticMethod::create(name, fn);
			m_underlying_type.__dict__->insert(String{ "__new__" }, new_fn);
		} else {
			m_underlying_type.__dict__->insert(
				String{ "__new__" }, std::get<PyObject *>(*m_underlying_type.__new__));
		}
	}
	for (auto method : m_underlying_type.__methods__) {
		auto *name = PyString::create(method.name);
		auto *method_fn = PyMethodDescriptor::create(name, this, method.method);
		m_underlying_type.__dict__->insert(String{ method.name }, method_fn);
	}

	if (!m_underlying_type.__bases__) {
		// not ideal, but avoids recursively calling object()
		if (m_underlying_type.__name__ == "object") {
			m_underlying_type.__bases__ = PyTuple::create();
		} else {
			m_underlying_type.__bases__ = PyTuple::create(object());
		}
	}
	m_underlying_type.__dict__->insert(String{ "__bases__" }, m_underlying_type.__bases__);
	// not ideal, but avoids recursively calling object()
	if (m_underlying_type.__name__ == "object") {
		m_underlying_type.__mro__ = PyTuple::create(this);
	} else {
		m_underlying_type.__mro__ = mro_internal();
	}
	m_underlying_type.__dict__->insert(String{ "__mro__" }, m_underlying_type.__mro__);

	update_methods_and_class_attributes(ns);
}

PyObject *PyType::new_(PyTuple *args, PyDict *kwargs) const
{
	if (std::holds_alternative<PyObject *>(*m_underlying_type.__new__)) {
		auto *obj = std::get<PyObject *>(*m_underlying_type.__new__);
		// prepend class type to args tuple -> obj->call((cls, *args), kwargs)
		std::vector<Value> args_with_type;
		args_with_type.reserve(args->size() + 1);
		// FIXME: remove this const_cast. Either args are const or PyType::new_ should not be const
		args_with_type.push_back(const_cast<PyType *>(this));
		for (const auto &el : args->elements()) { args_with_type.push_back(el); }
		args = PyTuple::create(args_with_type);
		return obj->call(args, kwargs);
	} else if (m_underlying_type.__new__.has_value()) {
		return std::get<NewSlotFunctionType>(*m_underlying_type.__new__)(this, args, kwargs);
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
	} else {
		// by default set object as a base if none provided
		bases = PyTuple::create(object());
	}

	return PyType::build_type(name, bases, ns);
}

PyType *PyType::build_type(PyString *type_name, PyTuple *bases, PyDict *ns)
{
	for (const auto &[key, value] : ns->map()) {
		const auto key_str = PyObject::from(key);
		if (!as<PyString>(key_str)) {
			VirtualMachine::the().interpreter().raise_exception(type_error(""));
		}
	}

	if (bases->elements().empty()) {
		// all objects inherit from object by default
		bases = PyTuple::create(object());
	}

	auto *base = PyObject::from(bases->elements()[0]);
	ASSERT(as<PyType>(base))

	auto *type = VirtualMachine::the().heap().allocate<PyType>(as<PyType>(base)->underlying_type());
	type->m_underlying_type.__name__ = type_name->value();
	type->m_underlying_type.__bases__ = bases;
	type->m_underlying_type.__mro__ = nullptr;
	type->initialize(ns);

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
			type_error("cannot create '{}' instances", m_type_prototype.__name__));
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
				TODO();
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

PyTuple *PyType::mro_internal() const
{
	if (!m_underlying_type.__mro__) {
		// FIXME: is this const_cast still needed?
		const auto &result = mro_(const_cast<PyType *>(this));
		m_underlying_type.__mro__ = PyTuple::create(result);
	}
	return m_underlying_type.__mro__;
}

PyList *PyType::mro() { return PyList::create(mro_internal()->elements()); }

bool PyType::issubclass(const PyType *other)
{
	if (this == other) { return true; }

	// every type is a subclass of object
	if (other == object()) { return true; }

	auto *this_mro = mro_internal();
	for (const auto &el : this_mro->elements()) {
		if (std::get<PyObject *>(el) == other) { return true; }
	}

	return false;
}

void PyType::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);

#define VISIT_SLOT(__slot__)                                                  \
	if (m_underlying_type.__slot__.has_value()                                \
		&& std::holds_alternative<PyObject *>(*m_underlying_type.__slot__)) { \
		visitor.visit(*std::get<PyObject *>(*m_underlying_type.__slot__));    \
	}

	VISIT_SLOT(__repr__)
	VISIT_SLOT(__call__)
	VISIT_SLOT(__new__)
	VISIT_SLOT(__init__)
	VISIT_SLOT(__hash__)
	VISIT_SLOT(__lt__)
	VISIT_SLOT(__le__)
	VISIT_SLOT(__eq__)
	VISIT_SLOT(__ne__)
	VISIT_SLOT(__gt__)
	VISIT_SLOT(__ge__)
	VISIT_SLOT(__iter__)
	VISIT_SLOT(__next__)
	VISIT_SLOT(__len__)
	VISIT_SLOT(__add__)
	VISIT_SLOT(__sub__)
	VISIT_SLOT(__mul__)
	VISIT_SLOT(__exp__)
	VISIT_SLOT(__lshift__)
	VISIT_SLOT(__mod__)
	VISIT_SLOT(__abs__)
	VISIT_SLOT(__neg__)
	VISIT_SLOT(__pos__)
	VISIT_SLOT(__invert__)
	VISIT_SLOT(__bool__)
	VISIT_SLOT(__getattribute__)
	VISIT_SLOT(__setattribute__)
	VISIT_SLOT(__get__)
#undef VISIT_SLOT

	if (m_underlying_type.__dict__) { visitor.visit(*m_underlying_type.__dict__); }

	if (m_underlying_type.__mro__) { visitor.visit(*m_underlying_type.__mro__); }

	if (m_underlying_type.__bases__) { visitor.visit(*m_underlying_type.__bases__); }

	if (m_underlying_type.__class__) { visitor.visit(*m_underlying_type.__class__); }
}