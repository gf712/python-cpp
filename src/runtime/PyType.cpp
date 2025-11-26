#include "PyType.hpp"
#include "AttributeError.hpp"
#include "PyBool.hpp"
#include "PyBoundMethod.hpp"
#include "PyBuiltInMethod.hpp"
#include "PyCell.hpp"
#include "PyClassMethodDescriptor.hpp"
#include "PyDict.hpp"
#include "PyFrame.hpp"
#include "PyFunction.hpp"
#include "PyGetSetDescriptor.hpp"
#include "PyInteger.hpp"
#include "PyList.hpp"
#include "PyMappingProxy.hpp"
#include "PyMemberDescriptor.hpp"
#include "PyMethodDescriptor.hpp"
#include "PyNone.hpp"
#include "PySlotWrapper.hpp"
#include "PyStaticMethod.hpp"
#include "PyString.hpp"
#include "StopIteration.hpp"
#include "TypeError.hpp"
#include "ValueError.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyTuple.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

#include <algorithm>
#include <string_view>
#include <unordered_set>

namespace py {
template<> PyType *as(PyObject *obj)
{
	if (obj->type()->underlying_type().is_type) { return static_cast<PyType *>(obj); }
	return nullptr;
}

template<> const PyType *as(const PyObject *obj)
{
	if (obj->type()->underlying_type().is_type) { return static_cast<const PyType *>(obj); }
	return nullptr;
}

// FIXME: copied from PyObject.cpp.
// Ideally this would live somewhere where all classes could see it, but this would require
// reorganising the project structure, since some of the functions used here, are not known
// when PyObject is declared in PyObject.hpp

namespace {
	template<typename SlotFunctionType,
		typename ResultType = typename SlotFunctionType::result_type,
		typename... Args>
	ResultType call_slot(const std::variant<SlotFunctionType, PyObject *> &slot, Args &&...args_)
		requires std::is_same_v<typename ResultType::OkType, PyObject *>
	{
		if (std::holds_alternative<SlotFunctionType>(slot)) {
			return std::get<SlotFunctionType>(slot)(std::forward<Args>(args_)...);
		} else if (std::holds_alternative<PyObject *>(slot)) {
			// FIXME: this const_cast is needed since in Python land there is no concept of
			//		  PyObject constness (right?). But for the internal calls handled above
			//		  which are resolved in the C++ runtime, we want to enforce constness
			//		  so we end up with the awkward line below. But how could we do better?
			auto args = PyTuple::create(const_cast<PyObject *>(
				static_cast<const PyObject *>(std::forward<Args>(args_)))...);
			if (args.is_err()) { return Err(args.unwrap_err()); }
			PyDict *kwargs = nullptr;
			return std::get<PyObject *>(slot)->call(args.unwrap(), kwargs);
		} else {
			TODO();
		}
	}

	template<typename SlotFunctionType,
		typename ResultType = typename SlotFunctionType::result_type,
		typename... Args>
	ResultType call_slot(const std::variant<SlotFunctionType, PyObject *> &slot,
		std::string_view conversion_error_message,
		Args &&...args_)
		requires(!std::is_same_v<typename ResultType::OkType, PyObject *>)
	{
		if (std::holds_alternative<SlotFunctionType>(slot)) {
			return std::get<SlotFunctionType>(slot)(std::forward<Args>(args_)...);
		} else if (std::holds_alternative<PyObject *>(slot)) {
			// FIXME: this const_cast is needed since in Python land there is no concept of
			//		  PyObject constness (right?). But for the internal calls handled above
			//		  which are resolved in the C++ runtime, we want to enforce constness
			//		  so we end up with the awkward line below. But how could we do better?
			auto args = PyTuple::create(const_cast<PyObject *>(
				static_cast<const PyObject *>(std::forward<Args>(args_)))...);
			if (args.is_err()) { return Err(args.unwrap_err()); }
			PyDict *kwargs = nullptr;
			if constexpr (std::is_same_v<typename ResultType::OkType, bool>) {
				auto result = std::get<PyObject *>(slot)->call(args.unwrap(), kwargs);
				if (result.is_err()) return Err(result.unwrap_err());
				if (!as<PyBool>(result.unwrap())) {
					return Err(type_error(std::string(conversion_error_message)));
				}
				return Ok(as<PyBool>(result.unwrap())->value());
			} else if constexpr (std::is_integral_v<typename ResultType::OkType>) {
				auto result = std::get<PyObject *>(slot)->call(args.unwrap(), kwargs);
				if (result.is_err()) return Err(result.unwrap_err());
				if (!as<PyInteger>(result.unwrap())) {
					return Err(type_error(std::string(conversion_error_message)));
				}
				return Ok(as<PyInteger>(result.unwrap())->as_i64());
			} else if constexpr (std::is_same_v<typename ResultType::OkType, std::monostate>) {
				auto result = std::get<PyObject *>(slot)->call(args.unwrap(), kwargs);
				if (result.is_err()) {
					return Err(result.unwrap_err());
				} else {
					return Ok(std::monostate{});
				}
			} else {
				[]<bool flag = false>() { static_assert(flag, "unsupported return type"); }();
			}
		} else {
			TODO();
		}
	}
}// namespace

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
	if (&type->underlying_type() == &types::BuiltinTypes::the().object()) { return { type }; }

	std::vector<PyObject *> mro_types;
	mro_types.push_back(type);

	std::vector<std::vector<PyObject *>> bases_mro;

	const auto &bases = type->underlying_type().__bases__;
	for (const auto &base : bases) {
		if (auto *precomputed_mro = base->__mro__) {
			std::vector<PyObject *> base_mro;
			base_mro.reserve(precomputed_mro->size());
			for (const auto &el : precomputed_mro->elements()) {
				base_mro.push_back(std::get<PyObject *>(el));
			}
			bases_mro.push_back(base_mro);
		} else {
			bases_mro.push_back(mro_(base));
		}
	}

	auto result = merge(bases_mro);
	mro_types.insert(mro_types.end(), result.begin(), result.end());

	return mro_types;
}

PyType::PyType(PyType *type)
	: PyBaseObject(type), m_underlying_type(type->underlying_type()),
	  m_metaclass(types::BuiltinTypes::the().type())
{}

PyType::PyType(TypePrototype &type_prototype)
	: PyBaseObject(types::BuiltinTypes::the().type()), m_underlying_type(type_prototype),
	  m_metaclass(types::BuiltinTypes::the().type())
{
	if (&type_prototype == &types::BuiltinTypes::the().type()) { underlying_type().is_type = true; }
}


PyType::PyType(std::unique_ptr<TypePrototype> &&type_prototype)
	: PyBaseObject(types::BuiltinTypes::the().type()), m_underlying_type(std::move(type_prototype)),
	  m_metaclass(types::BuiltinTypes::the().type())
{}

PyResult<PyType *> PyType::create(PyType *type)
{
	auto new_type_prototype = std::make_unique<TypePrototype>();
	auto *new_type = VirtualMachine::the().heap().allocate<PyType>(std::move(new_type_prototype));
	new_type->m_metaclass = type;
	if (!new_type) { return Err(memory_error(sizeof(PyType))); }
	return Ok(new_type);
}

std::string PyType::name() const
{
	auto index = underlying_type().__name__.find_last_of('.');
	if (index == std::string::npos) {
		return underlying_type().__name__;
	} else {
		return underlying_type().__name__.substr(index + 1);
	}
}

PyType *PyType::static_type() const
{
	if (&underlying_type() == &types::BuiltinTypes::the().type()) {
		return const_cast<PyType *>(this);// :(
	} else {
		if (std::holds_alternative<PyType *>(m_metaclass)) {
			return std::get<PyType *>(m_metaclass);
		} else {
			// all static types are of type `<class 'type'>`
			return types::type();
		}
	}
}

PyType *PyType::initialize(TypePrototype &type_prototype)
{
	auto *type = VirtualMachine::the().heap().allocate<PyType>(type_prototype);
	auto result = type->ready();
	ASSERT(result.is_ok());
	return type;
}

PyType *PyType::initialize(std::unique_ptr<TypePrototype> &&type_prototype)
{
	auto *type = VirtualMachine::the().heap().allocate<PyType>(std::move(type_prototype));
	auto result = type->ready();
	ASSERT(result.is_ok());
	return type;
}

namespace {
	std::once_flag type_flag;

	std::unique_ptr<TypePrototype> register_type_()
	{
		return std::move(
			klass<PyType>("type")
				.def("mro", &PyType::mro)
				.attr("__mro__", &PyType::__mro__)
				.classmethod("__prepare__",
					[](PyType *, PyTuple *, PyDict *) -> PyResult<PyObject *> {
						return PyDict::create();
					})
				.property(
					"__name__",
					[](PyType *self) { return PyString::create(self->name()); },
					[](PyObject *self, PyObject *value) -> PyResult<std::monostate> {
						(void)self;
						(void)value;
						TODO();
					})
				.property_readonly(
					"__dict__", [](PyType *self) { return PyMappingProxy::create(self->dict()); })
				.property_readonly("__bases__",
					[](PyType *self) {
						std::vector<PyObject *> bases;
						bases.insert(bases.begin(),
							self->underlying_type().__bases__.begin(),
							self->underlying_type().__bases__.end());
						return PyTuple::create(std::move(bases));
					})
				.property(
					"__abstractmethods__",
					[](PyType *self) -> PyResult<PyObject *> {
						if (self != types::type()) {
							if (auto result =
									(*self->attributes())[String{ "__abstractmethods__" }];
								result.has_value()) {
								return PyObject::from(*result);
							}
						}
						return Err(attribute_error("__abstractmethods__"));
					},
					[](PyType *self, PyObject *value) -> PyResult<std::monostate> {
						auto abstract = truthy(value, VirtualMachine::the().interpreter());
						if (abstract.is_err()) { return Err(abstract.unwrap_err()); }
						self->attributes()->insert(String{ "__abstractmethods__" }, value);
						return Ok(std::monostate{});
					})
				.type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyType::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(type_flag, []() { type = register_type_(); });
		return std::move(type);
	};
}

namespace {
	bool descriptor_is_data(const PyObject *obj)
	{
		return obj->type()->underlying_type().__set__.has_value();
	}
}// namespace

PyResult<PyObject *> PyType::__getattribute__(PyObject *attribute) const
{
	auto *name = as<PyString>(attribute);
	if (!name) {
		return Err(type_error("attribute name must be string, not '{}'",
			attribute->type()->underlying_type().__name__));
	}

	auto meta_attr_ = type()->lookup(name);
	if (meta_attr_.has_value() && meta_attr_->is_err()) { return *meta_attr_; }
	if (meta_attr_.has_value() && meta_attr_->is_ok()) {
		auto *meta_attr = meta_attr_->unwrap();
		const auto &meta_get = meta_attr->type()->underlying_type().__get__;
		if (meta_get.has_value() && descriptor_is_data(meta_attr)) {
			return call_slot(*meta_get, meta_attr, const_cast<PyType *>(this), type());
		}
	}

	auto attr_ = lookup(name);
	if (attr_.has_value()) {
		if (attr_->is_err()) { return *attr_; }
		auto *attr = attr_->unwrap();
		if (attr) {
			const auto &local_get = attr->type()->underlying_type().__get__;
			if (local_get.has_value()) {
				return call_slot(*local_get, attr, nullptr, const_cast<PyType *>(this));
			}
			return Ok(attr);
		}
	}

	if (meta_attr_.has_value() && meta_attr_->is_ok()) {
		auto *meta_attr = meta_attr_->unwrap();
		const auto &meta_get = meta_attr->type()->underlying_type().__get__;
		if (meta_get.has_value()) {
			return call_slot(*meta_get, meta_attr, const_cast<PyType *>(this), type());
		}
		return Ok(meta_attr);
	}

	return Err(attribute_error(
		"type object '{}' has no attribute '{}'", underlying_type().__name__, name->value()));
}

std::optional<PyResult<PyObject *>> PyType::lookup(PyObject *name) const
{
	auto mro = mro_internal();
	if (mro.is_err()) { return mro; }
	for (const auto &t_ : mro.unwrap()->elements()) {
		ASSERT(std::holds_alternative<PyObject *>(t_))
		auto *t = as<PyType>(std::get<PyObject *>(t_));
		ASSERT(t)
		ASSERT(t->underlying_type().__dict__)
		const auto &dict = t->underlying_type().__dict__->map();
		if (auto it = dict.find(name); it != dict.end()) { return PyObject::from(it->second); }
	}
	return std::nullopt;
}

PyResult<PyObject *> PyType::heap_object_allocation(PyType *type)
{
	return type->mro()
		.and_then([type](PyList *mro) -> PyResult<PyObject *> {
			for (const auto &el : mro->elements()) {
				ASSERT(std::holds_alternative<PyObject *>(el));
				ASSERT(as<PyType>(std::get<PyObject *>(el)));
				auto *t = as<PyType>(std::get<PyObject *>(el));
				if (!t->underlying_type().is_heaptype) {
					return t->underlying_type().__alloc__(type);
				}
			}
			return types::object()->underlying_type().__alloc__(type);
		})
		.and_then([](PyObject *obj) -> PyResult<PyObject *> {
			if (!obj->attributes()) {
				if (auto dict = PyDict::create(); dict.is_ok()) {
					obj->m_attributes = dict.unwrap();
				} else {
					return dict;
				}
			}
			return Ok(obj);
		});
}

PyResult<std::monostate> PyType::initialize(const std::string &name,
	PyType *base,
	std::vector<PyType *> bases,
	const PyDict *ns)
{
	auto dict_ = PyDict::create(ns->map());
	if (dict_.is_err()) { return Err(dict_.unwrap_err()); }
	auto *dict = dict_.unwrap();
	bool may_add_dict = base->attributes() == nullptr;
	if (auto it = dict->map().find(String{ "__slots__" }); it != dict->map().end()) {
		// has slots
		auto slots_ = PyObject::from(it->second);
		ASSERT(slots_.is_ok());
		auto *slots = slots_.unwrap();
		if (as<PyString>(slots)) {
			slots = PyTuple::create(slots).unwrap();
		} else {
			auto slots_list = PyList::create();
			if (slots_list.is_err()) { return Err(slots_list.unwrap_err()); }
			auto iter = slots->iter();
			if (iter.is_err()) { return Err(iter.unwrap_err()); }

			auto value_ = iter.unwrap()->next();
			while (value_.is_ok()) {
				slots_list.unwrap()->elements().push_back(value_.unwrap());
				value_ = iter.unwrap()->next();
			}

			if (!value_.unwrap_err()->type()->issubclass(types::stop_iteration())) {
				return Err(value_.unwrap_err());
			}

			slots = PyTuple::create(slots_list.unwrap()->elements()).unwrap();
		}
		ASSERT(as<PyTuple>(slots));

		const auto nslots = as<PyTuple>(slots)->size();

		for (const auto &el : as<PyTuple>(slots)->elements()) {
			auto slot_ = PyObject::from(el);
			ASSERT(slot_.is_ok());
			auto *slot = slot_.unwrap();
			ASSERT(as<PyString>(slot));
			if (as<PyString>(slot)->value() == "__dict__" && may_add_dict) {
				return Err(value_error("__dict__ slot disallowed: we already got one"));
			}
		}

		__slots__.reserve(nslots);
		for (const auto &el : as<PyTuple>(slots)->elements()) {
			__slots__.push_back(PyObject::from(el).unwrap());
		}

		// TODO: should we mangle and sort slots?
	}

	underlying_type().__name__ = name;
	underlying_type().__bases__ = std::move(bases);
	underlying_type().__base__ = base;
	underlying_type().basicsize = base->underlying_type().basicsize;

	underlying_type().__dict__ = dict;
	m_attributes = dict;

	if (!m_attributes->map().contains(String{ "__module__" })) {
		auto *globals = VirtualMachine::the().interpreter().execution_frame()->globals();
		if (auto *g = as<PyDict>(globals)) {
			if (auto it = g->map().find(String{ "__name__" }); it != g->map().end()) {
				m_attributes->insert(String{ "__module__" }, it->second);
			}
		} else {
			TODO();
		}
	}

	if (auto it = m_attributes->map().find(String{ "__qualname__" });
		it != m_attributes->map().end()) {
		auto qualname_ = PyObject::from(it->second);
		ASSERT(qualname_.is_ok());
		if (!as<PyString>(qualname_.unwrap())) {
			return Err(type_error(
				"type __qualname__ must be a str, not '{}'", qualname_.unwrap()->type()->name()));
		}
		__qualname__ = as<PyString>(qualname_.unwrap());
	} else {
		__qualname__ = PyString::create(underlying_type().__name__).unwrap();
	}

	if (auto it = m_attributes->map().find(String{ "__doc__" }); it != m_attributes->map().end()) {
		auto doc_ = PyObject::from(it->second);
		ASSERT(doc_.is_ok());
		if (auto doc_str = as<PyString>(doc_.unwrap())) {
			// unlike CPython we don't truncate if the string has a null terminator
			underlying_type().__doc__ = doc_str->value();
		}
	}

	if (auto it = m_attributes->map().find(String{ "__new__" }); it != m_attributes->map().end()) {
		auto new_slot_ = PyObject::from(it->second);
		ASSERT(new_slot_.is_ok());
		auto *new_slot = new_slot_.unwrap();
		if (auto fn = as<PyFunction>(new_slot)) {
			auto new_fn = PyStaticMethod::create(fn);
			ASSERT(new_fn.is_ok());
			new_slot = new_fn.unwrap();
			m_attributes->insert(String{ "__new__" }, new_slot);
		}
		underlying_type().__new__ = new_slot;
	}

	// TODO: uncomment when classmethod is fixed
	// if (auto it = m_attributes->map().find(String{ "__init_subclass__" });
	// 	it != m_attributes->map().end()) {
	// 	auto init_subclass_slot_ = PyObject::from(it->second);
	// 	ASSERT(init_subclass_slot_.is_ok());
	// 	auto *init_subclass_slot = init_subclass_slot_.unwrap();
	// 	if (auto fn = as<PyFunction>(init_subclass_slot)) {
	// 		auto new_fn = PyClassMethod::create(fn);
	// 		ASSERT(new_fn.is_ok());
	// 		underlying_type().__new__ = new_fn.unwrap();
	// 		m_attributes->insert(String{ "__init_subclass__" }, new_fn.unwrap());
	// 	}
	// }

	// TODO: uncomment when classmethod is fixed
	// if (auto it = m_attributes->map().find(String{ "___class_getitem__" });
	// 	it != m_attributes->map().end()) {
	// 	auto class_getitem_slot_ = PyObject::from(it->second);
	// 	ASSERT(class_getitem_slot_.is_ok());
	// 	auto *class_getitem_slot = class_getitem_slot_.unwrap();
	// 	if (auto fn = as<PyFunction>(class_getitem_slot)) {
	// 		auto new_fn = PyClassMethod::create(fn);
	// 		ASSERT(new_fn.is_ok());
	// 		underlying_type().__new__ = new_fn.unwrap();
	// 		m_attributes->insert(String{ "___class_getitem__" }, new_fn.unwrap());
	// 	}
	// }

	if (auto it = m_attributes->map().find(String{ "__classcell__" });
		it != m_attributes->map().end()) {
		auto classcell_ = PyObject::from(it->second);
		ASSERT(classcell_.is_ok());
		auto *classcell = classcell_.unwrap();
		if (auto *cell = as<PyCell>(classcell)) {
			cell->set_cell(this);
			auto r = m_attributes->delete_item(PyString::create("__classcell__").unwrap());
			if (r.is_err()) { return Err(r.unwrap_err()); }
		}
	}

	if (!__slots__.empty()) {
		size_t additional_offset = 0;
		for (const auto &slot : __slots__) {
			const auto name = as<PyString>(slot)->value();
			underlying_type().add_member(MemberDefinition{
				.name = name,
				.member_accessor = [additional_offset, name = std::move(name)](
									   PyObject *self) -> PyResult<PyObject *> {
					const auto object_offset =
						self->type()->underlying_type().basicsize + additional_offset;
					auto *obj = *bit_cast<PyObject **>(bit_cast<uint8_t *>(self) + object_offset);
					if (!obj) { return Err(attribute_error(name)); }
					return Ok(obj);
				},
				.member_setter = [additional_offset](
									 PyObject *self, PyObject *value) -> PyResult<std::monostate> {
					const auto offset =
						self->type()->underlying_type().basicsize + additional_offset;
					uint8_t *self_address = bit_cast<uint8_t *>(self);
					uint8_t *slot_address = self_address + offset;
					*bit_cast<PyObject **>(slot_address) = value;
					return Ok(std::monostate{});
				},
			});
			additional_offset += sizeof(PyObject *);
		}
	}

	underlying_type().__alloc__ = &PyType::heap_object_allocation;
	auto result = ready();
	if (result.is_err()) { return Err(result.unwrap_err()); }

	fixup_slots();

	return Ok(std::monostate{});
}

static std::array slotdefs = {
	Slot{ "__getattribute__", &TypePrototype::__getattribute__ },
	Slot{ "__getattr__", &TypePrototype::__getattribute__ },
	Slot{ "__setattr__", &TypePrototype::__setattribute__ },
	// Slot{"__delattr__", &TypePrototype::__delattribute__},
	Slot{ "__repr__", &TypePrototype::__repr__, "__repr__($self, /)\n--\n\nReturn repr(self)." },
	Slot{ "__hash__", &TypePrototype::__hash__, "__hash__($self, /)\n--\n\nReturn hash(self)." },
	Slot::with_keyword("__call__",
		&TypePrototype::__call__,
		"__call__($self, /, *args, **kwargs)\n--\n\nCall self as a function."),
	Slot{ "__str__", &TypePrototype::__str__, "__str__($self, /)\n--\n\nReturn str(self)." },
	Slot{ "__lt__", &TypePrototype::__lt__, "__lt__($self, value, /)\n--\n\nReturn self<value." },
	Slot{ "__le__", &TypePrototype::__le__, "__le__($self, value, /)\n--\n\nReturn self<=value." },
	Slot{ "__eq__", &TypePrototype::__eq__, "__eq__($self, value, /)\n--\n\nReturn self==value." },
	Slot{ "__ne__", &TypePrototype::__ne__, "__ne__($self, value, /)\n--\n\nReturn self!=value." },
	Slot{ "__gt__", &TypePrototype::__gt__, "__gt__($self, value, /)\n--\n\nReturn self>value." },
	Slot{ "__ge__", &TypePrototype::__ge__, "__ge__($self, value, /)\n--\n\nReturn self>=value." },
	Slot{ "__iter__", &TypePrototype::__iter__, "__iter__($self, /)\n--\n\nImplement iter(self)." },
	Slot{ "__next__", &TypePrototype::__next__, "__next__($self, /)\n--\n\nImplement next(self)." },
	Slot{ "__get__",
		&TypePrototype::__get__,
		"__get__($self, instance, owner, /)\n--\n\nReturn an attribute of instance, which is "
		"of "
		"type owner." },
	Slot{ "__set__",
		&TypePrototype::__set__,
		"__set__($self, instance, value, /)\n--\n\nSet an attribute of instance to value." },
	// Slot{ "__delete__",
	// 	&TypePrototype::__delete__,
	// 	"__delete__($self, instance, /)\n--\n\nDelete an attribute of instance." },
	Slot::with_keyword("__init__",
		&TypePrototype::__init__,
		"__init__($self, /, *args, **kwargs)\n--\n\n Initialize self.  See help(type(self)) "
		"for "
		"accurate signature."),
	Slot::with_new("__new__",
		&TypePrototype::__new__,
		"__new__(type, /, *args, **kwargs)\n--\n\n Create and return new object.  See "
		"help(type) "
		"for accurate signature."),
	// Slot{ "__del__", &TypePrototype::__del__ },

	// await
	// Slot{ "__await__",
	// 	&TypePrototype::__await__,
	// 	"__await__($self, /)\n--\n\nReturn an iterator to be used in await expression." },
	// Slot{ "__aiter__",
	// 	&TypePrototype::__aiter__,
	// 	"__aiter__($self, /)\n--\n\nReturn an awaitable, that resolves in asynchronous
	// iterator." }, Slot{ "__anext__", 	&TypePrototype::__anext__,
	// 	"__anext__($self, /)\n--\n\nReturn a value or raise StopAsyncIteration." },

	// number
	Slot{ "__add__",
		&TypePrototype::__add__,
		"__add__($self, value, /)\n--\n\nReturn self + value." },
	// Slot{ "__radd__",
	// 	&TypePrototype::__add__,
	// 	"__add__($self, value, /)\n--\n\nReturn value + self." },
	Slot{ "__sub__",
		&TypePrototype::__sub__,
		"__sub__($self, value, /)\n--\n\nReturn self - value." },
	// Slot{ "__rsub__",
	// 	&TypePrototype::__sub__,
	// 	"__rsub__($self, value, /)\n--\n\nReturn value - self." },
	Slot{ "__mul__",
		&TypePrototype::__mul__,
		"__mul__($self, value, /)\n--\n\nReturn self * value." },
	// Slot{ "__rmul__",
	// 	&TypePrototype::__rmul__,
	// 	"__rmul__($self, value, /)\n--\n\nReturn value * self." },
	Slot{ "__mod__",
		&TypePrototype::__mod__,
		"__mod__($self, value, /)\n--\n\nReturn self % value." },
	// Slot{ "__rmod__",
	// 	&TypePrototype::__rmod__,
	// 	"__rmod__($self, value, /)\n--\n\nReturn value % self." },
	// Slot{ "__divmod__", &TypePrototype::__divmod__, "Return divmod(self, value)." },
	// Slot{ "__rdivmod__", &TypePrototype::__rdivmod__, "Return divmod(value, self)." },
	Slot{ "__pow__",
		&TypePrototype::__pow__,
		"__pow__($self, value, mod=None, /)\n--\n\nReturn pow(self, value, mod)." },
	// Slot{ "__rpow__",
	// 	&TypePrototype::__rpow__,
	// 	"__pow__($self, value, mod=None, /)\n--\n\nReturn pow(value, self, mod)." },
	Slot{ "__neg__", &TypePrototype::__neg__, "__neg__($self, /)\n--\n\n-self" },
	Slot{ "__pos__", &TypePrototype::__pos__, "__pos__($self, /)\n--\n\n+self" },
	Slot{ "__abs__", &TypePrototype::__abs__, "__abs__($self, /)\n--\n\nabs(self)" },
	Slot{ "__bool__", &TypePrototype::__bool__, "__bool__($self, /)\n--\n\nself != 0" },
	Slot{ "__invert__", &TypePrototype::__invert__, "__invert__($self, /)\n--\n\n~self" },
	Slot{ "__lshift__",
		&TypePrototype::__lshift__,
		"__lshift__($self, value, /)\n--\n\nReturn self << value." },
	// Slot{ "__rlshift__",
	// 	&TypePrototype::__lshift__,
	// 	"__lshift__($self, value, /)\n--\n\nReturn value << self." },
	Slot{ "__rshift__",
		&TypePrototype::__rshift__,
		"__rshift__($self, value, /)\n--\n\nReturn self >> value." },
	// Slot{ "__rrshift__",
	// 	&TypePrototype::__rshift__,
	// 	"__rshift__($self, value, /)\n--\n\nReturn value >> self." },
	Slot{ "__and__",
		&TypePrototype::__and__,
		"__and__($self, value, /)\n--\n\nReturn self & value." },
	// Slot{ "__rand__",
	// 	&TypePrototype::__and__,
	// 	"__and__($self, value, /)\n--\n\nReturn value & self." },
	Slot{ "__xor__",
		&TypePrototype::__xor__,
		"__xor__F($self, value, /)\n--\n\nReturn self ^ value." },
	// Slot{ "__rxor__",
	// 	&TypePrototype::__xor__,
	// 	"__xor__($self, value, /)\n--\n\nReturn value ^ self." },
	Slot{ "__or__", &TypePrototype::__or__, "__or__($self, value, /)\n--\n\nReturn self | value." },
	// Slot{ "__ror__",
	// 	&TypePrototype::__or__,
	// 	"__or__($self, value, /)\n--\n\nReturn value | self." },
	// Slot{ "__int__", &TypePrototype::__int__, "__int__($self, /)\n--\n\n-int(self)" },
	// Slot{ "__float__", &TypePrototype::__float__, "__float__($self, /)\n--\n\n-float(self)"
	// }, Slot{ "__iadd__", 	&TypePrototype::__iadd__,
	// 	"__iadd__($self, value, /)\n--\n\nReturn self+=value." },
	// Slot{ "__isub__",
	// 	&TypePrototype::__isub__,
	// 	"__isub__($self, value, /)\n--\n\nReturn self-=value." },
	// Slot{ "__imul__",
	// 	&TypePrototype::__imul__,
	// 	"__imul__($self, value, /)\n--\n\nReturn self*=value." },
	// Slot{ "__imod__",
	// 	&TypePrototype::__imod__,
	// 	"__imod__($self, value, /)\n--\n\nReturn self%=value." },
	// Slot{ "__ipow__",
	// 	&TypePrototype::__ipow__,
	// 	"__ipow__($self, value, /)\n--\n\nReturn self**=value." },
	// Slot{ "__ilshift__",
	// 	&TypePrototype::__ilshift__,
	// 	"__ilshift__($self, value, /)\n--\n\nReturn self<<=value." },
	// Slot{ "__irshift__",
	// 	&TypePrototype::__irshift__,
	// 	"__irshift__($self, value, /)\n--\n\nReturn self>>=value." },
	// Slot{ "__iand__",
	// 	&TypePrototype::__iand__,
	// 	"__iand__($self, value, /)\n--\n\nReturn self&=value." },
	// Slot{ "__ixor__",
	// 	&TypePrototype::__ixor__,
	// 	"__ixor__($self, value, /)\n--\n\nReturn self^=value." },
	// Slot{ "__ior__",
	// 	&TypePrototype::__ior__,
	// 	"__ior__($self, value, /)\n--\n\nReturn self|=value." },
	Slot{ "__floordiv__",
		&TypePrototype::__floordiv__,
		"__floordiv__($self, value, /)\n--\n\nReturn self//value." },
	// Slot{ "__rfloordiv__",
	// 	&TypePrototype::__floordiv__,
	// 	"__floordiv__($self, value, /)\n--\n\nReturn value//self." },
	Slot{ "__truediv__",
		&TypePrototype::__truediv__,
		"__truediv__($self, value, /)\n--\n\nReturn self/value." },
	// Slot{ "__rtruediv__",
	// 	&TypePrototype::__truediv__,
	// 	"__truediv__($self, value, /)\n--\n\nReturn value/self." },
	// Slot{ "__ifloordiv__",
	// 	&TypePrototype::__ifloordiv__,
	// 	"__ifloordiv__($self, value, /)\n--\n\nReturn self//=value." },
	// Slot{ "__itruediv__",
	// 	&TypePrototype::__itruediv__,
	// 	"__itruediv__($self, value, /)\n--\n\nReturn self/=value." },
	// Slot{ "__index__",
	// 	&TypePrototype::__index__,
	// 	"__index__($self, /)\n--\n\n"
	// 	"Return self converted to an integer, if self is suitable for use as an index into a "
	// 	"list." },
	// Slot{ "__matmul__",
	// 	&TypePrototype::__matmul__,
	// 	"__matmul__($self, value, /)\n--\n\nReturn self@value." },
	// Slot{ "__rmatmul__",
	// 	&TypePrototype::__rmatmul__,
	// 	"__rmatmul__($self, value, /)\n--\n\nReturn value@self." },
	// Slot{ "__imatmul__",
	// 	&TypePrototype::__imatmul__,
	// 	"__imatmul__($self, value, /)\n--\n\nReturn value@=self." },

	// mapping
	Slot{ "__len__", &MappingTypePrototype::__len__, "__len__($self, /)\n--\n\nReturn len(self)." },
	Slot{ "__getitem__",
		&MappingTypePrototype::__getitem__,
		"__getitem__($self, key, /)\n--\n\nReturn self[key]." },
	Slot{ "__setitem__",
		&MappingTypePrototype::__setitem__,
		"__setitem__($self, key, value, /)\n--\n\nSet self[key] to value." },
	Slot{ "__delitem__",
		&MappingTypePrototype::__delitem__,
		"__delitem__($self, key, /)\n--\n\nDelete self[key]." },

	Slot{ "__len__",
		&SequenceTypePrototype::__len__,
		"__len__($self, /)\n--\n\nReturn len(self)." },
	Slot{ "__getitem__",
		&SequenceTypePrototype::__getitem__,
		"__getitem__($self, key, /)\n--\n\nReturn self[key]." },
	Slot{ "__setitem__",
		&SequenceTypePrototype::__setitem__,
		"__setitem__($self, key, value, /)\n--\n\nSet self[key] to value." },
	Slot{ "__delitem__",
		&SequenceTypePrototype::__delitem__,
		"__delitem__($self, key, /)\n--\n\nDelete self[key]." },
	Slot{ "__contains__",
		&SequenceTypePrototype::__contains__,
		"__contains__($self, key, /)\n--\n\nReturn key in self." },

	// Slot{ "__add__",
	// 	&SequenceTypePrototype::__concat__,
	// 	"__add__($self, value, /)\n--\n\nReturn self+value." },

	// SQSLOT("__mul__", sq_repeat, NULL, wrap_indexargfunc,
	//        "__mul__($self, value, /)\n--\n\nReturn self*value."),
	// SQSLOT("__rmul__", sq_repeat, NULL, wrap_indexargfunc,
	//        "__rmul__($self, value, /)\n--\n\nReturn value*self."),
	// SQSLOT("__iadd__", sq_inplace_concat, NULL,
	//        wrap_binaryfunc,
	//        "__iadd__($self, value, /)\n--\n\nImplement self+=value."),
	// SQSLOT("__imul__", sq_inplace_repeat, NULL,
	//        wrap_indexargfunc,
	//        "__imul__($self, value, /)\n--\n\nImplement self*=value."),
};

namespace {
	PyResult<PyObject *> new_wrapper(PyObject *self, PyTuple *args, PyDict *kwargs)
	{
		auto *type = as<PyType>(self);
		if (!type) {
			// TODO: should be SystemError, not ValueError
			return Err(value_error("__new__() called with non-type 'self'"));
		}
		if (!args || args->elements().empty()) {
			return Err(type_error("{}.__new__(): not enough arguments", type->name()));
		}
		auto arg0 = PyObject::from(args->elements()[0]);
		if (arg0.is_err()) return arg0;
		auto subtype = as<PyType>(arg0.unwrap());
		if (!subtype) {
			return Err(type_error("{}.__new__(X): X is not a type object ({})",
				type->name(),
				arg0.unwrap()->type()->name()));
		}
		if (!subtype->issubclass(type)) {
			return Err(type_error("{}.__new__({}): {} is not a subtype of {}",
				type->name(),
				subtype->name(),
				subtype->name(),
				type->name()));
		}

		// check that the most derived base that's not a heap type is this type.
		auto staticbase = subtype;
		while (staticbase && staticbase->underlying_type().is_heaptype) {
			staticbase = staticbase->underlying_type().__base__;
		}
		auto same_slot = []<typename SlotFnType>(
							 std::optional<std::variant<SlotFnType, PyObject *>> &lhs,
							 std::variant<SlotFnType, PyObject *> rhs) -> bool {
			ASSERT(std::holds_alternative<SlotFnType>(rhs));
			if (!lhs.has_value()) { return false; }
			if (std::holds_alternative<PyObject *>(*lhs)) { return false; }
			return get_address(rhs) == get_address(*lhs);
		};
		ASSERT(type->underlying_type().__new__.has_value());
		if (staticbase
			&& !same_slot(
				staticbase->underlying_type().__new__, *type->underlying_type().__new__)) {
			return Err(type_error("{}.__new__({}) is not safe, use {}.__new__()",
				type->name(),
				subtype->name(),
				staticbase->name()));
		}

		std::vector<Value> new_args_vec;
		new_args_vec.insert(
			new_args_vec.end(), args->elements().begin() + 1, args->elements().end());
		auto new_args = PyTuple::create(std::move(new_args_vec));
		if (new_args.is_err()) return new_args;
		auto &new_slot = *type->underlying_type().__new__;
		ASSERT(std::holds_alternative<NewSlotFunctionType>(new_slot));
		return std::get<NewSlotFunctionType>(new_slot)(subtype, new_args.unwrap(), kwargs);
	}
}// namespace

PyResult<std::monostate> PyType::add_operators()
{
	for (auto &&slot : slotdefs) {
		if (slot.name == "__new__") { continue; }
		if (auto it = m_attributes->map().find(String{ std::string{ slot.name } });
			it != m_attributes->map().end()) {
			auto fn = PyObject::from(it->second);
			if (fn.is_err()) return Err(fn.unwrap_err());
			slot.update_member(underlying_type(), fn.unwrap());
		} else if (!slot.has_member(underlying_type())) {
			continue;
		} else {
			auto name = PyString::create(std::string{ slot.name });
			if (name.is_err()) { return Err(name.unwrap_err()); }
			auto descr = slot.create_slot_wrapper(this);
			if (descr.is_err()) return Err(descr.unwrap_err());
			m_attributes->insert(String{ std::string{ slot.name } }, descr.unwrap());
		}
	}

	if (underlying_type().__new__.has_value()) {
		if (!m_attributes->map().contains(String{ "__new__" })) {
			auto new_fn_obj = PyNativeFunction::create("__new__", new_wrapper, this);
			if (new_fn_obj.is_err()) return Err(new_fn_obj.unwrap_err());
			m_attributes->insert(String{ "__new__" }, new_fn_obj.unwrap());
		}
	}

	return Ok(std::monostate{});
}

PyResult<std::monostate> PyType::add_methods()
{
	if (underlying_type().__methods__.empty()) { return Ok(std::monostate{}); }
	ASSERT(m_attributes);
	for (auto &method : underlying_type().__methods__) {
		auto [name, fn, flags, doc] = method;
		auto name_str_ = PyString::create(name);
		if (name_str_.is_err()) { return Err(name_str_.unwrap_err()); }
		auto *name_str = name_str_.unwrap();
		auto descriptor = [&, flags = flags]() -> PyResult<PyObject *> {
			if (flags.is_set(MethodFlags::Flag::CLASSMETHOD)) {
				return PyClassMethodDescriptor::create(name_str, this, method);
			} else if (flags.is_set(MethodFlags::Flag::STATICMETHOD)) {
				return PyBuiltInMethod::create(method, this);
			} else {
				return PyMethodDescriptor::create(name_str, this, method);
			}
		}();
		if (descriptor.is_err()) { return Err(descriptor.unwrap_err()); }

		// TODO: Could this check be done before constructing the descriptor?
		if (m_attributes->map().contains(String{ name })) { continue; }

		m_attributes->insert(String{ name }, descriptor.unwrap());
	}

	return Ok(std::monostate{});
}

PyResult<std::monostate> PyType::add_members()
{
	if (underlying_type().__members__.empty()) { return Ok(std::monostate{}); }
	ASSERT(m_attributes);
	for (auto &member : underlying_type().__members__) {
		auto [name, accessor, setter] = member;
		auto name_str_ = PyString::create(name);
		if (name_str_.is_err()) { return Err(name_str_.unwrap_err()); }
		auto *name_str = name_str_.unwrap();
		auto descriptor = PyMemberDescriptor::create(name_str, this, accessor, setter);
		if (descriptor.is_err()) { return Err(descriptor.unwrap_err()); }

		// TODO: Could this check be done before constructing the descriptor?
		if (m_attributes->map().contains(String{ name })) { continue; }

		m_attributes->insert(String{ name }, descriptor.unwrap());
	}
	return Ok(std::monostate{});
}

PyResult<std::monostate> PyType::add_properties()
{
	if (underlying_type().__getset__.empty()) { return Ok(std::monostate{}); }
	ASSERT(m_attributes);
	for (auto &getset : underlying_type().__getset__) {
		auto [name, accessor, setter] = getset;

		auto name_str_ = PyString::create(name);
		if (name_str_.is_err()) { return Err(name_str_.unwrap_err()); }
		auto *name_str = name_str_.unwrap();
		auto descriptor = PyGetSetDescriptor::create(name_str, this, getset);
		if (descriptor.is_err()) { return Err(descriptor.unwrap_err()); }

		// TODO: Could this check be done before constructing the descriptor?
		if (m_attributes->map().contains(String{ name })) { continue; }

		m_attributes->insert(String{ name }, descriptor.unwrap());
	}
	return Ok(std::monostate{});
}

void PyType::inherit_special(PyType *base)
{
	if (&base->underlying_type() != &types::BuiltinTypes::the().object()
		|| underlying_type().is_heaptype) {
		if (!underlying_type().__new__.has_value()) {
			underlying_type().__new__ = base->underlying_type().__new__;
		}
	}

	if (base->issubtype(types::BuiltinTypes::the().type())) { underlying_type().is_type = true; }
}

PyResult<std::monostate> PyType::inherit_slots(PyType *base)
{
	auto *basebase = base->underlying_type().__base__;
	auto slot_defined =
		[base, basebase]<typename FnType>(
			std::function<std::optional<std::variant<FnType, PyObject *>>(TypePrototype &)>
				get_slot) {
			auto base_slot = get_slot(base->underlying_type());
			const bool base_has_slot = base_slot.has_value();
			const bool basebase_exists = basebase != nullptr;
			if (!basebase_exists) { return base_has_slot; }
			auto basebase_slot = get_slot(basebase->underlying_type());
			bool base_and_basebase_slots_notequal = true;
			if (basebase_slot.has_value() && base_slot.has_value()) {
				if (std::holds_alternative<PyObject *>(*base_slot)
					&& std::holds_alternative<PyObject *>(*basebase_slot)) {
					base_and_basebase_slots_notequal =
						std::get<PyObject *>(*base_slot) != std::get<PyObject *>(*basebase_slot);
				}
				if (std::holds_alternative<FnType>(*base_slot)
					&& std::holds_alternative<FnType>(*basebase_slot)) {
					base_and_basebase_slots_notequal =
						&std::get<FnType>(*base_slot) != &std::get<FnType>(*basebase_slot);
				}
			}

			return base_has_slot && (basebase_exists || base_and_basebase_slots_notequal);
		};

	auto add_slot = [this, base, slot_defined]<typename FnType>(
						std::optional<std::variant<FnType, PyObject *>> TypePrototype::*slot_ptr) {
		auto base_slot = base->underlying_type().*slot_ptr;
		std::function<std::optional<std::variant<FnType, PyObject *>>(TypePrototype &)> get_slot =
			[slot_ptr](TypePrototype &t) -> std::optional<std::variant<FnType, PyObject *>> {
			return t.*slot_ptr;
		};
		if (!get_slot(this->underlying_type()).has_value() && slot_defined(get_slot)) {
			auto &this_slot = this->underlying_type().*slot_ptr;
			this_slot = base_slot;
		}
	};

	// FIXME: should use number_type_protocol
	auto add_numeric_slot = add_slot;

	auto add_sequence_slot = [this, base, slot_defined]<typename FnType>(
								 std::optional<std::variant<FnType, PyObject *>>
									 SequenceTypePrototype::*slot_ptr) {
		if (!this->underlying_type().sequence_type_protocol.has_value()) return;
		if (!base->underlying_type().sequence_type_protocol.has_value()) return;
		auto base_slot = (*base->underlying_type().sequence_type_protocol).*slot_ptr;
		std::function<std::optional<std::variant<FnType, PyObject *>>(TypePrototype &)> get_slot =
			[slot_ptr](TypePrototype &t) -> std::optional<std::variant<FnType, PyObject *>> {
			if (!t.sequence_type_protocol.has_value()) { return std::nullopt; }
			return (*t.sequence_type_protocol).*slot_ptr;
		};
		if (!get_slot(this->underlying_type()).has_value() && slot_defined(get_slot)) {
			auto &this_slot = (*this->underlying_type().sequence_type_protocol).*slot_ptr;
			this_slot = base_slot;
		}
	};

	auto add_mapping_slot = [this, base, slot_defined]<typename FnType>(
								std::optional<std::variant<FnType, PyObject *>>
									MappingTypePrototype::*slot_ptr) {
		if (!this->underlying_type().mapping_type_protocol.has_value()) return;
		if (!base->underlying_type().mapping_type_protocol.has_value()) return;
		auto base_slot = (*base->underlying_type().mapping_type_protocol).*slot_ptr;
		std::function<std::optional<std::variant<FnType, PyObject *>>(TypePrototype &)> get_slot =
			[slot_ptr](TypePrototype &t) -> std::optional<std::variant<FnType, PyObject *>> {
			if (!t.mapping_type_protocol.has_value()) { return std::nullopt; }
			return (*t.mapping_type_protocol).*slot_ptr;
		};
		if (!get_slot(this->underlying_type()).has_value() && slot_defined(get_slot)) {
			auto &this_slot = (*this->underlying_type().mapping_type_protocol).*slot_ptr;
			this_slot = base_slot;
		}
	};

	// auto add_buffer_slot = [this, base, slot_defined]<typename FnType>(
	// 						   std::optional<std::variant<FnType, PyObject *>> PyBufferProcs::*
	// 							   slot_ptr) {
	// 	if (!this->underlying_type().as_buffer.has_value()) return;
	// 	if (!base->underlying_type().as_buffer.has_value()) return;
	// 	auto base_slot = (*base->underlying_type().as_buffer).*slot_ptr;
	// 	std::function<std::optional<std::variant<FnType, PyObject *>>(TypePrototype &)> get_slot =
	// 		[slot_ptr](TypePrototype &t) -> std::optional<std::variant<FnType, PyObject *>> {
	// 		if (!t.as_buffer.has_value()) { return std::nullopt; }
	// 		return (*t.as_buffer).*slot_ptr;
	// 	};
	// 	if (slot_defined(get_slot)) {
	// 		auto &this_slot = (*this->underlying_type().as_buffer).*slot_ptr;
	// 		this_slot = base_slot;
	// 	}
	// };

	add_numeric_slot(&TypePrototype::__add__);
	add_numeric_slot(&TypePrototype::__sub__);
	add_numeric_slot(&TypePrototype::__mul__);
	add_numeric_slot(&TypePrototype::__mod__);
	// add_numeric_slot(&TypePrototype::__divmod__);
	// add_numeric_slot(&TypePrototype::__pow__);
	add_numeric_slot(&TypePrototype::__neg__);
	add_numeric_slot(&TypePrototype::__pos__);
	add_numeric_slot(&TypePrototype::__abs__);
	add_numeric_slot(&TypePrototype::__bool__);
	add_numeric_slot(&TypePrototype::__invert__);
	add_numeric_slot(&TypePrototype::__lshift__);
	// add_numeric_slot(&TypePrototype::__rshift__);
	add_numeric_slot(&TypePrototype::__and__);
	// add_numeric_slot(&TypePrototype::__xor__);
	add_numeric_slot(&TypePrototype::__or__);
	// add_numeric_slot(&TypePrototype::__int__);
	// add_numeric_slot(&TypePrototype::__float__);
	// add_numeric_slot(&TypePrototype::__iadd__);
	// add_numeric_slot(&TypePrototype::__isub__);
	// add_numeric_slot(&TypePrototype::__imul__);
	// add_numeric_slot(&TypePrototype::__imod__);
	// add_numeric_slot(&TypePrototype::__ipow__);
	// add_numeric_slot(&TypePrototype::__ilshift__);
	// add_numeric_slot(&TypePrototype::__irshift__);
	// add_numeric_slot(&TypePrototype::__iand__);
	// add_numeric_slot(&TypePrototype::__ixor__);
	// add_numeric_slot(&TypePrototype::__ior__);
	// add_numeric_slot(&TypePrototype::__truediv__);
	// add_numeric_slot(&TypePrototype::__floordiv__);
	// add_numeric_slot(&TypePrototype::__index__);
	// add_numeric_slot(&TypePrototype::__matmul__ );
	// add_numeric_slot(&TypePrototype::__imatmul__);
	add_sequence_slot(&SequenceTypePrototype::__len__);
	add_sequence_slot(&SequenceTypePrototype::__concat__);
	add_sequence_slot(&SequenceTypePrototype::__getitem__);
	add_sequence_slot(&SequenceTypePrototype::__setitem__);
	add_sequence_slot(&SequenceTypePrototype::__delitem__);
	add_sequence_slot(&SequenceTypePrototype::__contains__);

	add_mapping_slot(&MappingTypePrototype::__len__);
	add_mapping_slot(&MappingTypePrototype::__getitem__);
	add_mapping_slot(&MappingTypePrototype::__setitem__);
	add_mapping_slot(&MappingTypePrototype::__delitem__);

	// add_buffer_slot(&PyBufferProcs::getbuffer);
	// add_buffer_slot(&PyBufferProcs::releasebuffer);

	add_slot(&TypePrototype::__getattribute__);
	add_slot(&TypePrototype::__setattribute__);
	add_slot(&TypePrototype::__repr__);
	add_slot(&TypePrototype::__call__);
	add_slot(&TypePrototype::__str__);
	add_slot(&TypePrototype::__eq__);
	add_slot(&TypePrototype::__gt__);
	add_slot(&TypePrototype::__ge__);
	add_slot(&TypePrototype::__le__);
	add_slot(&TypePrototype::__lt__);
	add_slot(&TypePrototype::__ne__);
	add_slot(&TypePrototype::__hash__);
	add_slot(&TypePrototype::__iter__);
	add_slot(&TypePrototype::__next__);
	add_slot(&TypePrototype::__get__);
	add_slot(&TypePrototype::__set__);
	add_slot(&TypePrototype::__init__);

	return Ok(std::monostate{});
}


PyResult<PyObject *> PyType::new_(PyTuple *args, PyDict *kwargs) const
{
	if (underlying_type().__new__.has_value()) {
		if (std::holds_alternative<PyObject *>(*underlying_type().__new__)) {
			auto *obj = std::get<PyObject *>(*underlying_type().__new__);
			auto fn = obj->get(const_cast<PyType *>(this), nullptr);
			return fn.and_then([this, args, kwargs](PyObject *constructor) -> PyResult<PyObject *> {
				// pop out type from args
				std::vector<Value> new_args;
				new_args.reserve(args->size() + 1);
				new_args.push_back(const_cast<PyType *>(this));
				new_args.insert(new_args.end(), args->elements().begin(), args->elements().end());
				auto args_ = PyTuple::create(new_args);
				if (args_.is_err()) { return args_; }
				return constructor->call(args_.unwrap(), kwargs);
			});
		} else if (std::holds_alternative<NewSlotFunctionType>(*underlying_type().__new__)) {
			auto fn = std::get<NewSlotFunctionType>(*underlying_type().__new__);
			ASSERT(fn);
			return fn(this, args, kwargs);
		} else {
			TODO();
		}
	} else {
		return Err(type_error("cannot create '{}' instances", underlying_type().__name__));
	}
}

PyResult<std::monostate> PyType::ready()
{
	if (underlying_type().is_ready) { return Ok(std::monostate{}); }

	if (underlying_type().__name__.empty()) {
		// FIXME: should return system error
		return Err(type_error("Type does not define the __name__ field."));
	}

	auto *base = underlying_type().__base__;
	// default base is `object`, unless this is already `object`
	if (!base && &underlying_type() != &types::BuiltinTypes::the().object()) {
		base = types::object();
		underlying_type().__base__ = types::object();
	}

	// initialize base class
	if (base) { base->ready(); }

	// initialize bases
	auto bases = underlying_type().__bases__;
	if (bases.empty() && &underlying_type() != &types::BuiltinTypes::the().object()) {
		if (base) { underlying_type().__bases__ = { base }; }
	}
	// initialize dict
	if (!underlying_type().__dict__) {
		auto dict = PyDict::create();
		if (dict.is_err()) return Err(dict.unwrap_err());
		underlying_type().__dict__ = dict.unwrap();
	}
	m_attributes = underlying_type().__dict__;

	if (auto result = add_operators(); result.is_err()) { return result; }
	if (auto result = add_methods(); result.is_err()) { return result; }
	if (auto result = add_members(); result.is_err()) { return result; }
	if (auto result = add_properties(); result.is_err()) { return result; }

	if (auto result = mro_internal(); result.is_err()) { return Err(result.unwrap_err()); }

	if (underlying_type().__base__) { inherit_special(underlying_type().__base__); }

	ASSERT(__mro__);

	for (size_t i = 1; i < __mro__->elements().size(); ++i) {
		const auto &b = __mro__->elements()[i];
		auto base_ = PyObject::from(b);
		ASSERT(base_.is_ok());
		auto *base = base_.unwrap();

		// type(base) == type
		if ((std::holds_alternative<PyType *>(m_type)
				&& std::get<PyType *>(m_type) == types::type())
			|| (&std::get<std::reference_wrapper<const TypePrototype>>(m_type).get()
				== &types::BuiltinTypes::the().type())) {
			inherit_slots(static_cast<PyType *>(base));
		}
	}

	if (underlying_type().__doc__.has_value()) {
		m_attributes->insert(
			String{ "__doc__" }, String{ std::string{ *underlying_type().__doc__ } });
	} else {
		m_attributes->insert(String{ "__doc__" }, py_none());
	}

	if (!underlying_type().__hash__.has_value()) {
		if (auto it = m_attributes->map().find(String{ "__hash__" });
			it != m_attributes->map().end()) {
			auto hash_fn_ = PyObject::from(it->second);
			if (hash_fn_.is_err()) return Err(hash_fn_.unwrap_err());
			underlying_type().__hash__ = hash_fn_.unwrap();
		} else {
			m_attributes->insert(String{ "__hash__" }, py_none());
		}
	}

	if (auto *base = underlying_type().__base__) {
		if (!underlying_type().number_type_protocol.has_value()) {
			underlying_type().number_type_protocol = base->underlying_type().number_type_protocol;
		}
		if (!underlying_type().sequence_type_protocol.has_value()) {
			underlying_type().sequence_type_protocol =
				base->underlying_type().sequence_type_protocol;
		}
		if (!underlying_type().mapping_type_protocol.has_value()) {
			underlying_type().mapping_type_protocol = base->underlying_type().mapping_type_protocol;
		}
		if (!underlying_type().as_buffer.has_value()) {
			underlying_type().as_buffer = base->underlying_type().as_buffer;
		}
	}

	// TODO: add subclass

	// Done!
	underlying_type().is_ready = true;

	return Ok(std::monostate{});
}

namespace {
	std::optional<std::reference_wrapper<Slot>> resolve_slotdups(PyType *type,
		std::string_view name)
	{
		std::vector<std::reference_wrapper<Slot>> slots;
		for (auto &slotdef : slotdefs) {
			if (slotdef.name == name) { slots.emplace_back(slotdef); }
		}

		std::optional<std::reference_wrapper<Slot>> res;
		for (auto &slotdef : slots) {
			if (!slotdef.get().has_member(type->underlying_type())) {
				continue;
			} else if (res.has_value()) {
				return std::nullopt;
			}
			res = slotdef;
		}

		return res;
	}

	void update_slot(PyType *type, Slot &slot)
	{
		// First of all, if the slot in question does not exist, return immediately.
		if (!slot.has_member(type->underlying_type())) { return; }
		ASSERT(type->__mro__);
		std::optional<PyObject *> descr_;
		// For the given slot, we loop over all the special methods with a name corresponding to
		// that slot and we look up these names in the MRO of the type.
		for (const auto &b : type->__mro__->elements()) {
			auto base_ = PyObject::from(b);
			ASSERT(base_.is_ok());
			auto *base = base_.unwrap();
			auto *base_astype = as<PyType>(base);
			ASSERT(base_astype);
			auto *dict = base_astype->attributes();
			ASSERT(dict);
			if (auto it = dict->map().find(String{ std::string{ slot.name } });
				it != dict->map().end()) {
				descr_ = PyObject::from(it->second).unwrap();
				break;
			}
		}
		// If we don't find any special method, the slot is set to NULL (regardless of what was
		// in the slot before).
		if (!descr_.has_value()) {
			slot.reset_member(type->underlying_type());
			return;
		}

		std::optional<std::variant<void *, PyObject *>> generic;
		std::optional<std::variant<void *, PyObject *>> specific;
		bool use_generic = false;
		auto *descr = *descr_;
		ASSERT(descr);
		if (auto slot_wrapper = as<PySlotWrapper>(descr)) {
			auto slotdef = resolve_slotdups(type, slot.name);
			if (!slotdef.has_value()) {
				generic = PyNativeFunction::create(
					std::string{ slot.name },
					[slot_wrapper](PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
						std::vector<Value> new_args_vector;
						new_args_vector.reserve(args->size() - 1);
						auto self_ = PyObject::from(args->elements()[0]);
						if (self_.is_err()) return self_;
						auto *self = self_.unwrap();
						for (size_t i = 1; i < args->size(); ++i) {
							new_args_vector.push_back(args->elements()[i]);
						}
						auto args_ = PyTuple::create(new_args_vector);
						if (args_.is_err()) return args_;
						args = args_.unwrap();
						return slot_wrapper->slot()(self, args, kwargs);
					},
					slot_wrapper)
							  .unwrap();
				use_generic = true;
			} else if (auto s = slot.get_member(slot_wrapper->base_type()->underlying_type());
					   !s.has_value()) {
				// if we don't have a specific slot we get the wrapper from descriptor we found
				// in the MRO
				generic = slot_wrapper->base().get().get_member(
					slot_wrapper->base_type()->underlying_type());
				use_generic = true;
			} else {
				specific = *s;
				use_generic = false;
			}
		} else if (auto fn = as<PyNativeFunction>(descr);
				   fn && (fn->method_pointer().has_value() && *fn->method_pointer() == &new_wrapper)
				   && slot.name == "__new__") {
			auto &new_slot = type->underlying_type().__new__;
			ASSERT(new_slot.has_value());
			if (std::holds_alternative<NewSlotFunctionType>(*new_slot)) {
				using NewSlotFunctionPointerType =
					PyResult<PyObject *> (*)(const PyType *, PyTuple *, PyDict *);
				ASSERT(std::get<NewSlotFunctionType>(*new_slot));
				auto fn_ptr =
					std::get<NewSlotFunctionType>(*new_slot).target<NewSlotFunctionPointerType>();
				ASSERT(fn_ptr);
				specific = reinterpret_cast<void *>(*fn_ptr);
			} else {
				ASSERT(std::get<PyObject *>(*new_slot));
				specific = std::get<PyObject *>(*new_slot);
			}
		} else if (descr == py_none() && slot.name == "__hash__") {
			TODO();
		} else {
			use_generic = true;
			generic = slot.get_member(type->underlying_type());
		}

		if (specific.has_value() && !use_generic) {
			slot.set_member(type->underlying_type(), *specific);
			ASSERT(slot.get_member(type->underlying_type()));
		} else {
			ASSERT(generic.has_value());
			slot.set_member(type->underlying_type(), *generic);
			ASSERT(slot.get_member(type->underlying_type()));
		}
	}
}// namespace

void PyType::fixup_slots()
{
	for (auto &&slot : slotdefs) { update_slot(this, slot); }
}

PyResult<PyObject *> PyType::__new__(const PyType *type_, PyTuple *args, PyDict *kwargs)
{
	ASSERT(args && args->size() == 3)
	ASSERT(!kwargs || kwargs->map().empty())

	auto *name = as<PyString>(PyObject::from(args->elements()[0]).unwrap());
	ASSERT(name);
	auto *bases = as<PyTuple>(PyObject::from(args->elements()[1]).unwrap());
	ASSERT(bases);
	auto *ns = PyObject::from(args->elements()[2]).unwrap();

	std::vector<PyType *> bases_vector;
	for (const auto &base : bases->elements()) {
		auto base_ = PyObject::from(base);
		if (base_.is_err()) { return base_; }
		if (auto *b = as<PyType>(base_.unwrap())) {
			bases_vector.push_back(b);
		} else {
			return Err(type_error("bases must be types"));
		}
	}

	auto bases_result = compute_bases(type_, std::move(bases_vector), args, kwargs);
	if (bases_result.is_err()) { return Err(bases_result.unwrap_err()); }

	if (std::holds_alternative<PyObject *>(bases_result.unwrap())) {
		// metaclass is not `type`, so we called it's new implementation
		return Ok(std::get<PyObject *>(bases_result.unwrap()));
	}
	auto [base, bases_] = std::get<BasePair>(bases_result.unwrap());
	bases_vector = std::move(bases_);

	if (!ns->type()->issubclass(types::dict())) {
		return Err(
			type_error("type.__new__() argument 3 must be dict, not '{}'", ns->type()->name()));
	}

	return PyType::build_type(
		type_, name, base, std::move(bases_vector), static_cast<const PyDict *>(ns));
}

PyResult<PyType *> PyType::best_base(const std::vector<PyType *> &bases)
{
	// FIXME: find the "solid base" (https://peps.python.org/pep-0253/#multiple-inheritance)
	return Ok(bases[0]);
}

PyResult<std::variant<PyType::BasePair, PyObject *>> PyType::compute_bases(const PyType *type_,
	std::vector<PyType *> bases,
	PyTuple *args,
	PyDict *kwargs)
{
	if (!bases.empty()) {
		std::unordered_set<PyObject *> bases_set;
		for (const auto &base : bases) {
			if (bases_set.contains(base)) {
				auto *duplicate_type = as<PyType>(base);
				return Err(type_error(
					"duplicate base class {}", duplicate_type->underlying_type().__name__));
			}
			bases_set.insert(base);
			continue;
			if (auto [attr, r] =
					base->lookup_attribute(PyString::create("__mro_entries__").unwrap());
				attr.is_err() || r == LookupAttrResult::NOT_FOUND) {
				if (attr.is_err()) return attr;
				return Err(type_error(
					"type() doesn't support MRO entry resolution; use types.new_class()"));
			}
		}

		auto winner_ = calculate_metaclass(type_, bases);
		if (winner_.is_err()) { return Err(winner_.unwrap_err()); }
		auto *winner = winner_.unwrap();

		if (winner != type_) {
			if (get_address(*winner->type()->underlying_type().__new__)
				== get_address(*types::type()->underlying_type().__new__)) {
				return call_slot(*winner->type()->underlying_type().__new__, winner, args, kwargs);
			}
			type_ = winner;
		}

		auto base = best_base(bases);
		if (base.is_err()) { return Err(base.unwrap_err()); }
		return Ok(std::make_tuple(base.unwrap(), bases));
	}
	// by default set object as a base if none provided
	return Ok(std::make_tuple(types::object(), std::vector<PyType *>{ types::object() }));
}

PyResult<PyType *> PyType::build_type(const PyType *metatype,
	PyString *type_name,
	PyType *base,
	std::vector<PyType *> bases,
	const PyDict *ns)
{
	ASSERT(!bases.empty());

	for (const auto &[key, _] : ns->map()) {
		auto key_ = PyObject::from(key);
		if (key_.is_err()) { return Err(key_.unwrap_err()); }
		if (!as<PyString>(key_.unwrap())) {
			return Err(type_error("namespace key is of type '{}', but 'str' is required",
				key_.unwrap()->type()->name()));
		}
	}

	return PyType::create(const_cast<PyType *>(metatype)).and_then([&](PyType *type) {
		type->underlying_type().is_heaptype = true;
		type->__mro__ = nullptr;
		type->initialize(type_name->value(), base, std::move(bases), ns);

		spdlog::trace("Created type@{} #{}", (void *)type, type->name());

		return Ok(type);
	});
}

PyResult<const PyType *> PyType::calculate_metaclass(const PyType *type_,
	const std::vector<PyType *> &bases)
{
	auto *winner = type_;
	for (const auto &base : bases) {
		if (winner->issubclass(base->type())) { continue; }
		if (base->type()->issubclass(winner)) {
			winner = base->type();
			continue;
		}
		return Err(
			type_error("metaclass conflict: the metaclass of a derived class must be a "
					   "(non-strict) subclass of the metaclasses of all its bases"));
	}
	return Ok(winner);
}

PyResult<PyObject *> PyType::__call__(PyTuple *args, PyDict *kwargs) const
{
	if (this == types::type()) {
		if (args->size() == 1) {
			auto obj = PyObject::from(args->elements()[0]);
			if (obj.is_err()) return obj;
			return Ok(obj.unwrap()->type());
		}
		if (args->size() != 3) {
			return Err(type_error("type() takes 1 or 3 arguments, got {}", args->size()));
		}
	}

	auto obj_ = new_(args, kwargs);
	if (obj_.is_err()) { return obj_; }

	// If __new__() does not return an instance of cls, then the new instance’s __init__()
	// method will not be invoked.
	if (auto *obj = obj_.unwrap(); obj->type() == this) {
		// If __new__() is invoked during object construction and it returns an instance of cls,
		// then the new instance’s __init__() method will be invoked like __init__(self[, ...]),
		// where self is the new instance and the remaining arguments are the same as were
		// passed to the object constructor.
		if (const auto res = obj->init(args, kwargs); res.is_ok()) {
			if (res.unwrap() < 0) {
				// error
				TODO();
				return Err(nullptr);
			}
		} else {
			return Err(res.unwrap_err());
		}
	}
	return obj_;
}

std::string PyType::to_string() const
{
	return fmt::format("<class '{}'>", underlying_type().__name__);
}

PyResult<PyObject *> PyType::__repr__() const { return PyString::create(to_string()); }

PyResult<PyTuple *> PyType::mro_internal() const
{
	if (!__mro__) {
		// FIXME: is this const_cast still needed?
		const auto &result = mro_(const_cast<PyType *>(this));
		auto mro = PyTuple::create(result);
		if (mro.is_err()) { return mro; }
		__mro__ = mro.unwrap();
	}
	return Ok(__mro__);
}

PyResult<PyList *> PyType::mro()
{
	auto mro = mro_internal();
	if (mro.is_err()) { return Err(mro.unwrap_err()); }
	return PyList::create(mro.unwrap()->elements());
}


bool PyType::issubtype(const TypePrototype &other) const
{
	if (&underlying_type() == &other) { return true; }

	// every type is a subclass of object
	if (&other == &types::BuiltinTypes::the().object()) { return true; }

	// avoids creating PyTuple
	const auto this_mro = mro_(const_cast<PyType *>(this));
	for (const auto &el : this_mro) {
		// avoid calling PyObject::type, which may call `py::type()`, resulting in recursion
		if (&static_cast<PyType *>(el)->underlying_type() == &other) { return true; }
	}

	return false;
}

bool PyType::issubclass(const PyType *other) const
{
	if (this == other) { return true; }

	// every type is a subclass of object
	if (other == types::object()) { return true; }

	auto this_mro = mro_internal();
	if (this_mro.is_err()) { TODO(); }
	for (const auto &el : this_mro.unwrap()->elements()) {
		if (std::get<PyObject *>(el) == other) { return true; }
	}

	return false;
}

void PyType::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);

#define VISIT_SLOT(__slot__)                                                  \
	if (underlying_type().__slot__.has_value()                                \
		&& std::holds_alternative<PyObject *>(*underlying_type().__slot__)) { \
		visitor.visit(*std::get<PyObject *>(*underlying_type().__slot__));    \
	}

#define VISIT_MAPPING_SLOT(__slot__)                                                              \
	if (underlying_type().mapping_type_protocol.has_value()                                       \
		&& underlying_type().mapping_type_protocol->__slot__.has_value()                          \
		&& std::holds_alternative<PyObject *>(                                                    \
			*underlying_type().mapping_type_protocol->__slot__)) {                                \
		visitor.visit(*std::get<PyObject *>(*underlying_type().mapping_type_protocol->__slot__)); \
	}

#define VISIT_SEQUENCE_SLOT(__slot__)                                                              \
	if (underlying_type().sequence_type_protocol.has_value()                                       \
		&& underlying_type().sequence_type_protocol->__slot__.has_value()                          \
		&& std::holds_alternative<PyObject *>(                                                     \
			*underlying_type().sequence_type_protocol->__slot__)) {                                \
		visitor.visit(*std::get<PyObject *>(*underlying_type().sequence_type_protocol->__slot__)); \
	}

	VISIT_SLOT(__new__)
	VISIT_SLOT(__init__)

	VISIT_SLOT(__getattribute__)
	VISIT_SLOT(__setattribute__)
	VISIT_SLOT(__get__)
	VISIT_SLOT(__set__)

	VISIT_SLOT(__add__)
	VISIT_SLOT(__sub__)
	VISIT_SLOT(__mul__)
	VISIT_SLOT(__pow__)
	VISIT_SLOT(__lshift__)
	VISIT_SLOT(__mod__)
	VISIT_SLOT(__and__)
	VISIT_SLOT(__or__)
	VISIT_SLOT(__abs__)
	VISIT_SLOT(__neg__)
	VISIT_SLOT(__pos__)
	VISIT_SLOT(__invert__)

	VISIT_SLOT(__call__)
	VISIT_SLOT(__str__)
	VISIT_SLOT(__bool__)
	VISIT_SLOT(__repr__)
	VISIT_SLOT(__iter__)
	VISIT_SLOT(__next__)
	VISIT_SLOT(__hash__)

	VISIT_SLOT(__eq__)
	VISIT_SLOT(__gt__)
	VISIT_SLOT(__ge__)
	VISIT_SLOT(__le__)
	VISIT_SLOT(__lt__)
	VISIT_SLOT(__ne__)

	VISIT_MAPPING_SLOT(__len__)
	VISIT_MAPPING_SLOT(__setitem__)
	VISIT_MAPPING_SLOT(__getitem__)
	VISIT_MAPPING_SLOT(__delitem__)

	VISIT_SEQUENCE_SLOT(__len__)
	VISIT_SEQUENCE_SLOT(__concat__)
	VISIT_SEQUENCE_SLOT(__setitem__)
	VISIT_SEQUENCE_SLOT(__getitem__)
	VISIT_SEQUENCE_SLOT(__delitem__)
	VISIT_SEQUENCE_SLOT(__contains__)
#undef VISIT_SLOT

	underlying_type().visit_graph(visitor);

	if (__name__) { visitor.visit(*__name__); }
	if (__qualname__) { visitor.visit(*__qualname__); }
	if (__module__) { visitor.visit(*__module__); }
	if (__mro__) { visitor.visit(*__mro__); }
	if (std::holds_alternative<PyType *>(m_metaclass) && std::get<PyType *>(m_metaclass)) {
		visitor.visit(*std::get<PyType *>(m_metaclass));
	}

	for (auto *el : __slots__) { visitor.visit(*el); }
}
}// namespace py
