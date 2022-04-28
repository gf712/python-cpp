#include "PyType.hpp"
#include "PyBoundMethod.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyInteger.hpp"
#include "PyList.hpp"
#include "PyMemberDescriptor.hpp"
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

namespace py {
template<> PyType *as(PyObject *obj)
{
	if (obj->type() == py::type()) { return static_cast<PyType *>(obj); }
	return nullptr;
}

template<> const PyType *as(const PyObject *obj)
{
	if (obj->type() == py::type()) { return static_cast<const PyType *>(obj); }
	return nullptr;
}

// FIXME: copied from PyObject.cpp.
// Ideally this would live somewhere where all classes could see it, but this would require
// reorganising the project structure, since some of the functions used here, are not known
// when PyObject is declared in PyObject.hpp
template<typename SlotFunctionType, typename... Args>
static PyResult call_slot(const std::variant<SlotFunctionType, PyObject *> &slot, Args &&... args_)
{
	if (std::holds_alternative<SlotFunctionType>(slot)) {
		auto result = std::get<SlotFunctionType>(slot)(std::forward<Args>(args_)...);
		using ResultType = std::remove_pointer_t<decltype(result)>;

		if constexpr (std::is_same_v<PyResult, ResultType>) {
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
		auto args = PyTuple::create(
			const_cast<PyObject *>(static_cast<const PyObject *>(std::forward<Args>(args_)))...);
		if (args.is_err()) { return args; }
		PyDict *kwargs = nullptr;
		return std::get<PyObject *>(slot)->call(
			static_cast<PyTuple *>(args.template unwrap_as<PyObject>()), kwargs);
	} else {
		TODO();
	}
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
		return py::type();
	}
}

PyType *PyType::initialize(TypePrototype type_prototype)
{
	auto *type = VirtualMachine::the().heap().allocate_static<PyType>(type_prototype).get();
	type->initialize(nullptr);
	return type;
}

namespace {
	std::once_flag type_flag;

	std::unique_ptr<TypePrototype> register_type_()
	{
		return std::move(klass<PyType>("type").def("mro", &PyType::mro).type);
	}
}// namespace

std::unique_ptr<TypePrototype> PyType::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(type_flag, []() { type = register_type_(); });
	return std::move(type);
}

namespace {
	bool descriptor_is_data(const PyObject *obj)
	{
		// FIXME: temporary hack to get object.__new__ working, but requires __set__ to be
		// implemented
		//        should be:
		//        obj->type()->underlying_type().__set__.has_value()
		return !as<PyStaticMethod>(obj) && !as<PySlotWrapper>(obj);
	}
}// namespace

PyResult PyType::__getattribute__(PyObject *attribute) const
{
	auto *name = as<PyString>(attribute);
	if (!name) {
		return PyResult::Err(type_error("attribute name must be string, not '{}'",
			attribute->type()->underlying_type().__name__));
	}

	auto meta_attr_ = type()->lookup(name);
	if (meta_attr_.is_err()) { return meta_attr_; }
	auto *meta_attr = meta_attr_.unwrap_as<PyObject>();
	if (meta_attr) {
		const auto &meta_get = meta_attr->type()->underlying_type().__get__;
		if (meta_get.has_value() && descriptor_is_data(meta_attr)) {
			return call_slot(*meta_get, meta_attr, const_cast<PyType *>(this), type());
		}
	}

	auto attr_ = lookup(name);
	if (attr_.is_err()) { return attr_; }
	auto *attr = attr_.unwrap_as<PyObject>();
	if (attr) {
		const auto &local_get = attr->type()->underlying_type().__get__;
		if (local_get.has_value()) {
			return call_slot(*local_get, attr, nullptr, const_cast<PyType *>(this));
		}
		return PyResult::Ok(attr);
	}

	if (meta_attr) { return PyResult::Ok(meta_attr); }

	return PyResult::Err(type_error(
		"type object '{}' has no attribute '{}'", m_underlying_type.__name__, name->value()));
}

PyResult PyType::lookup(PyObject *name) const
{
	auto mro = mro_internal();
	if (mro.is_err()) { return mro; }
	for (const auto &t_ : static_cast<PyList *>(mro.unwrap_as<PyObject>())->elements()) {
		ASSERT(std::holds_alternative<PyObject *>(t_))
		auto *t = as<PyType>(std::get<PyObject *>(t_));
		ASSERT(t)
		ASSERT(t->m_underlying_type.__dict__)
		const auto &dict = t->m_underlying_type.__dict__->map();
		if (auto it = dict.find(name); it != dict.end()) { return PyObject::from(it->second); }
	}
	// TODO: should this return an error?
	return PyResult::Ok(py_none());
}

bool PyType::update_if_special(const std::string &name, const Value &value)
{
	if (!name.starts_with("__")) { return false; }
	auto obj_ = PyObject::from(value);
	if (obj_.is_err()) { TODO(); }
	auto *obj = obj_.unwrap_as<PyObject>();

	if (name == "__new__") {
		m_underlying_type.__new__ = obj;
		return true;
	}
	if (name == "__init__") {
		m_underlying_type.__init__ = obj;
		return true;
	}
	if (name == "__repr__") {
		m_underlying_type.__repr__ = obj;
		return true;
	}
	return false;
}

void PyType::update_methods_and_class_attributes(PyDict *ns)
{
	if (ns) {
		for (const auto &[key, v] : ns->map()) {
			auto attr_method_name = [](const auto &k) {
				if (std::holds_alternative<String>(k)) {
					return PyString::create(std::get<String>(k).s);
				} else if (std::holds_alternative<PyObject *>(k)) {
					auto *obj = std::get<PyObject *>(k);
					ASSERT(as<PyString>(obj))
					return PyResult::Ok(as<PyString>(obj));
				} else {
					TODO();
				}
			}(key);

			if (attr_method_name.is_err()) {
				// TODO: return error
				TODO();
			}

			update_if_special(
				static_cast<PyString *>(attr_method_name.unwrap_as<PyObject>())->value(), v);
			m_underlying_type.__dict__->insert(
				static_cast<PyString *>(attr_method_name.unwrap_as<PyObject>()), v);
		}
	}
}

namespace {
	template<typename SlotFunctionType, typename FunctorType>
	std::pair<String, PyObject *> wrap_slot(PyType *type,
		std::string_view name_sv,
		PyDict *ns,
		std::variant<SlotFunctionType, PyObject *> &slot,
		FunctorType &&f)
	{
		String name_str{ std::string(name_sv) };
		if (ns) {
			if (auto it = ns->map().find(name_str); it != ns->map().end()) {
				auto slot_ = PyObject::from(it->second);
				ASSERT(slot_.is_ok())
				return { name_str, slot_.template unwrap_as<PyObject>() };
			}
		}

		if (std::holds_alternative<SlotFunctionType>(slot)) {
			// FIXME: should PyString have a std::string_view constructor and not worry about the
			// 		  lifetime of the string_view?
			auto name_ = PyString::create(std::string(name_sv));
			if (name_.is_err()) { TODO(); }
			auto *name = static_cast<PyString *>(name_.template unwrap_as<PyObject>());
			// the lifetime of the type is extended by the slot wrapper
			auto func_ = PySlotWrapper::create(name, type, std::move(f));
			if (func_.is_err()) { TODO(); }
			auto *func = static_cast<PySlotWrapper *>(func_.template unwrap_as<PyObject>());
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
	auto dict = PyDict::create();
	if (dict.is_err()) { TODO(); }
	m_underlying_type.__dict__ = static_cast<PyDict *>(std::get<PyObject *>(dict.unwrap()));
	// m_attributes should be a "mappingproxy" object, not dict for PyType
	m_attributes = m_underlying_type.__dict__;
	if (m_underlying_type.__add__.has_value()) {
		auto [name, add_func] = wrap_slot(this,
			"__add__",
			ns,
			*m_underlying_type.__add__,
			[this](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult {
				ASSERT(args && args->size() == 1)
				ASSERT(!kwargs || kwargs->map().empty())
				auto arg0 = PyObject::from(args->elements()[0]);
				if (arg0.is_err()) return arg0;
				return std::get<AddSlotFunctionType>(*m_underlying_type.__add__)(
					self, arg0.unwrap_as<PyObject>());
			});
		m_underlying_type.__dict__->insert(name, add_func);
	}
	if (m_underlying_type.__repr__.has_value()) {
		auto [name, repr_func] = wrap_slot(this,
			"__repr__",
			ns,
			*m_underlying_type.__repr__,
			[this](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult {
				if (args && args->size() > 0) {
					return PyResult::Err(type_error("expected 0 arguments, got {}", args->size()));
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
			[this](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult {
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
			[this](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult {
				auto result =
					std::get<InitSlotFunctionType>(*m_underlying_type.__init__)(self, args, kwargs);
				if (result) { ASSERT(*result == 0) }
				return PyResult::Ok(py_none());
			});
		m_underlying_type.__dict__->insert(name, init_func);
	}
	if (m_underlying_type.__new__.has_value()) {
		auto name = PyString::create("__new__");
		if (name.is_err()) { TODO(); }
		if (std::holds_alternative<NewSlotFunctionType>(*m_underlying_type.__new__)) {
			auto fn = PyNativeFunction::create(
				"__new__",
				[this](PyTuple *args_, PyDict *kwargs) -> PyResult {
					auto maybe_type = PyObject::from(args_->elements()[0]);
					ASSERT(maybe_type.is_ok())
					auto *type = as<PyType>(maybe_type.unwrap_as<PyObject>());
					ASSERT(type)
					// pop out type from args
					std::vector<Value> new_args;
					new_args.reserve(args_->size() - 1);
					new_args.insert(
						new_args.end(), args_->elements().begin() + 1, args_->elements().end());
					auto args = PyTuple::create(new_args);
					if (args.is_err()) { return args; }
					return std::get<NewSlotFunctionType>(*m_underlying_type.__new__)(
						type, args.unwrap_as<PyTuple>(), kwargs);
				},
				this);
			if (fn.is_err()) { TODO(); }
			auto new_fn =
				PyStaticMethod::create(static_cast<PyString *>(name.unwrap_as<PyObject>()),
					static_cast<PyNativeFunction *>(fn.unwrap_as<PyObject>()));
			ASSERT(new_fn.is_ok())
			m_underlying_type.__dict__->insert(String{ "__new__" }, new_fn.unwrap_as<PyObject>());
		} else {
			m_underlying_type.__dict__->insert(
				String{ "__new__" }, std::get<PyObject *>(*m_underlying_type.__new__));
		}
	}
	for (auto member : m_underlying_type.__members__) {
		auto name = PyString::create(member.name);
		if (name.is_err()) { TODO(); }
		auto m = PyMemberDescriptor::create(
			static_cast<PyString *>(name.unwrap_as<PyObject>()), this, member.member_accessor);
		ASSERT(m.is_ok())
		m_underlying_type.__dict__->insert(String{ member.name }, m.unwrap_as<PyObject>());
	}
	for (auto method : m_underlying_type.__methods__) {
		auto name = PyString::create(method.name);
		if (name.is_err()) { TODO(); }
		auto method_fn = PyMethodDescriptor::create(
			static_cast<PyString *>(name.unwrap_as<PyObject>()), this, std::move(method.method));
		ASSERT(method_fn.is_ok())
		m_underlying_type.__dict__->insert(String{ method.name }, method_fn.unwrap_as<PyObject>());
	}

	if (!m_underlying_type.__bases__) {
		// not ideal, but avoids recursively calling object()
		if (m_underlying_type.__name__ == "object") {
			auto bases = PyTuple::create();
			if (bases.is_err()) { TODO(); }
			m_underlying_type.__bases__ = static_cast<PyTuple *>(bases.unwrap_as<PyObject>());
		} else {
			auto bases = PyTuple::create(object());
			if (bases.is_err()) { TODO(); }
			m_underlying_type.__bases__ = static_cast<PyTuple *>(bases.unwrap_as<PyObject>());
		}
	}
	m_underlying_type.__dict__->insert(String{ "__bases__" }, m_underlying_type.__bases__);
	// not ideal, but avoids recursively calling object()
	if (m_underlying_type.__name__ == "object") {
		auto mro = PyTuple::create(this);
		if (mro.is_err()) { TODO(); }
		m_underlying_type.__mro__ = static_cast<PyTuple *>(mro.unwrap_as<PyObject>());
	} else {
		auto mro = mro_internal();
		if (mro.is_err()) { TODO(); }
		m_underlying_type.__mro__ = static_cast<PyTuple *>(mro.unwrap_as<PyObject>());
	}
	m_underlying_type.__dict__->insert(String{ "__mro__" }, m_underlying_type.__mro__);

	update_methods_and_class_attributes(ns);
}

PyResult PyType::new_(PyTuple *args_, PyDict *kwargs) const
{
	if (std::holds_alternative<PyObject *>(*m_underlying_type.__new__)) {
		auto *obj = std::get<PyObject *>(*m_underlying_type.__new__);
		// prepend class type to args tuple -> obj->call((cls, *args), kwargs)
		std::vector<Value> args_with_type;
		args_with_type.reserve(args_->size() + 1);
		// FIXME: remove this const_cast. Either args are const or PyType::new_ should not be const
		args_with_type.push_back(const_cast<PyType *>(this));
		for (const auto &el : args_->elements()) { args_with_type.push_back(el); }
		auto args = PyTuple::create(args_with_type);
		if (args.is_err()) { return args; }
		return obj->call(static_cast<PyTuple *>(args.unwrap_as<PyObject>()), kwargs);
	} else if (m_underlying_type.__new__.has_value()) {
		return std::get<NewSlotFunctionType>(*m_underlying_type.__new__)(this, args_, kwargs);
	}
	TODO();
	return PyResult::Err(nullptr);
}


PyResult PyType::__new__(const PyType *type_, PyTuple *args, PyDict *kwargs)
{
	(void)type_;
	ASSERT(args && args->size() == 3)
	ASSERT(!kwargs || kwargs->map().empty())

	auto *name = as<PyString>(PyObject::from(args->elements()[0]).unwrap_as<PyObject>());
	ASSERT(name)
	auto *bases = as<PyTuple>(PyObject::from(args->elements()[1]).unwrap_as<PyObject>());
	ASSERT(bases)
	auto *ns = as<PyDict>(PyObject::from(args->elements()[2]).unwrap_as<PyObject>());
	ASSERT(ns)

	if (!bases->elements().empty()) {
		std::unordered_set<PyObject *> bases_set;
		for (const auto &b : bases->elements()) {
			ASSERT(std::holds_alternative<PyObject *>(b))
			if (bases_set.contains(std::get<PyObject *>(b))) {
				auto *duplicate_type = as<PyType>(std::get<PyObject *>(b));
				return PyResult::Err(type_error(
					"duplicate base class {}", duplicate_type->underlying_type().__name__));
			}
			bases_set.insert(std::get<PyObject *>(b));
		}
	} else {
		// by default set object as a base if none provided
		auto bases_ = PyTuple::create(object());
		if (bases_.is_err()) { return bases_; }
		bases = static_cast<PyTuple *>(bases_.unwrap_as<PyObject>());
	}

	return PyType::build_type(name, bases, ns);
}

PyResult PyType::build_type(PyString *type_name, PyTuple *bases, PyDict *ns)
{
	for (const auto &[key, value] : ns->map()) {
		const auto key_str = PyObject::from(key);
		if (key_str.is_err()) return key_str;
		if (!key_str.unwrap_as<PyString>()) { return PyResult::Err(type_error("")); }
	}

	if (bases->elements().empty()) {
		// all objects inherit from object by default
		auto bases_ = PyTuple::create(object());
		if (bases_.is_err()) { return bases_; }
		bases = static_cast<PyTuple *>(bases_.unwrap_as<PyObject>());
	}

	auto base_ = PyObject::from(bases->elements()[0]);
	if (base_.is_err()) return base_;
	auto base = base_.unwrap_as<PyObject>();
	ASSERT(as<PyType>(base))

	auto *type = VirtualMachine::the().heap().allocate<PyType>(as<PyType>(base)->underlying_type());
	if (!type) { return PyResult::Err(memory_error(sizeof(PyType))); }
	type->m_underlying_type.__name__ = type_name->value();
	type->m_underlying_type.__bases__ = bases;
	type->m_underlying_type.__mro__ = nullptr;
	type->initialize(ns);

	return PyResult::Ok(type);
}

PyResult PyType::__call__(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->map().size() == 0)
	if (this == py::type()) {
		if (args->size() == 1) {
			auto obj = PyObject::from(args->elements()[0]);
			if (obj.is_err()) return obj;
			return PyResult::Ok(obj.unwrap_as<PyObject>()->type());
		}
		if (args->size() != 3) {
			return PyResult::Err(type_error("type() takes 1 or 3 arguments, got {}", args->size()));
		}
	}

	auto obj_ = new_(args, kwargs);
	if (obj_.is_err()) {
		return PyResult::Err(type_error("cannot create '{}' instances", m_type_prototype.__name__));
	}

	// If __new__() does not return an instance of cls, then the new instance’s __init__() method
	// will not be invoked.
	if (auto *obj = obj_.unwrap_as<PyObject>(); obj->type() == this) {
		// If __new__() is invoked during object construction and it returns an instance of cls,
		// then the new instance’s __init__() method will be invoked like __init__(self[, ...]),
		// where self is the new instance and the remaining arguments are the same as were passed to
		// the object constructor.
		if (const auto res = obj->init(args, kwargs); res.has_value()) {
			if (*res < 0) {
				// error
				TODO();
				return PyResult::Err(nullptr);
			}
		}
	}
	return obj_;
}

std::string PyType::to_string() const
{
	return fmt::format("<class '{}'>", m_underlying_type.__name__);
}

PyResult PyType::__repr__() const { return PyString::create(to_string()); }

PyResult PyType::mro_internal() const
{
	if (!m_underlying_type.__mro__) {
		// FIXME: is this const_cast still needed?
		const auto &result = mro_(const_cast<PyType *>(this));
		auto mro = PyTuple::create(result);
		if (mro.is_err()) { return mro; }
		m_underlying_type.__mro__ = static_cast<PyTuple *>(mro.unwrap_as<PyObject>());
	}
	return PyResult::Ok(m_underlying_type.__mro__);
}

PyResult PyType::mro()
{
	auto mro = mro_internal();
	if (mro.is_err()) { return mro; }
	return PyList::create(static_cast<PyList *>(mro.unwrap_as<PyObject>())->elements());
}

bool PyType::issubclass(const PyType *other)
{
	if (this == other) { return true; }

	// every type is a subclass of object
	if (other == object()) { return true; }

	auto this_mro = mro_internal();
	if (this_mro.is_err()) { TODO(); }
	for (const auto &el : static_cast<PyList *>(this_mro.unwrap_as<PyObject>())->elements()) {
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
}// namespace py