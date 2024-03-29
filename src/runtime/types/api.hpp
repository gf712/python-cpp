#pragma once

#include "builtin.hpp"
#include "runtime/PyArgParser.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "runtime/TypeError.hpp"

namespace py {

PyObject *py_none();

template<typename T> struct klass
{
	std::unique_ptr<TypePrototype> type;
	PyModule *m_module;

	klass(PyModule *module, std::string_view name)
		: type(TypePrototype::create<T>(name)), m_module(module)
	{}

	template<typename... BaseType>
		requires(std::is_same_v<std::remove_reference_t<BaseType>, PyType *> && ...)
	klass(PyModule *module, std::string_view name, BaseType &&...bases)
		: type(TypePrototype::create<T>(name)), m_module(module)
	{
		type->__bases__ = std::vector<PyType *>{ bases... };
	}

	klass(std::string_view name) : type(TypePrototype::create<T>(name)) {}

	template<typename... BaseType>
		requires(std::is_same_v<std::remove_reference_t<BaseType>, PyType *> && ...)
	klass(std::string_view name, BaseType &&...bases) : type(TypePrototype::create<T>(name))
	{
		type->__bases__ = std::vector<PyType *>{ bases... };
	}

	using AttributeGetterType = std::function<PyResult<PyObject *>(T *)>;
	using AttributeSetterType = std::function<PyResult<std::monostate>(T *, PyObject *)>;

	klass &property(std::string_view name,
		std::optional<AttributeGetterType> &&getter,
		std::optional<AttributeSetterType> &&setter)
	{
		type->add_property(PropertyDefinition{
			.name = std::string(name),
			.member_getter = [getter = std::move(getter)]()
				-> std::optional<std::function<PyResult<PyObject *>(PyObject *)>> {
				if (getter.has_value()) {
					return [getter_ = std::move(getter)](PyObject *self) -> PyResult<PyObject *> {
						if (auto obj = getter_->operator()(static_cast<T *>(self));
							obj.is_ok() && !obj.unwrap()) {
							return Ok(py_none());
						} else {
							return obj;
						}
					};
				} else {
					return std::nullopt;
				}
			}(),
			.member_setter = [setter = std::move(setter)]()
				-> std::optional<std::function<PyResult<std::monostate>(PyObject *, PyObject *)>> {
				if (setter.has_value()) {
					return [setter_ = std::move(setter)](
							   PyObject *self, PyObject *value) -> PyResult<std::monostate> {
						return setter_->operator()(static_cast<T *>(self), value);
					};
				} else {
					return std::nullopt;
				}
			}(),
		});
		return *this;
	}

	klass &property_readonly(std::string_view name, std::optional<AttributeGetterType> &&getter)
	{
		return property(name, std::move(getter), {});
	}

	template<typename ClassMemberType>
	klass &attr(std::string_view name, ClassMemberType &&member)
		requires requires(PyObject *self) { static_cast<T *>(self)->*member; }
	{
		type->add_member(MemberDefinition{
			.name = std::string(name),
			.member_accessor = [member](PyObject *self) -> PyResult<PyObject *> {
				static_assert(member_pointer<ClassMemberType>{});
				using MemberType =
					typename std::remove_cvref_t<typename member_pointer<ClassMemberType>::type>;

				if constexpr (std::is_convertible_v<MemberType, PyObject *>) {
					PyObject *obj = static_cast<T *>(self)->*member;
					if (!obj) { return Ok(py_none()); }
					return Ok(obj);
				} else if constexpr (std::numeric_limits<MemberType>::is_integer) {
					if constexpr (std::numeric_limits<MemberType>::is_signed) {
						return PyInteger::create(
							static_cast<int64_t>(static_cast<T *>(self)->*member));
					} else {
						const auto value = static_cast<uint64_t>(static_cast<T *>(self)->*member);
						if (value > std::numeric_limits<int64_t>::max()) { TODO(); }
						return PyInteger::create(static_cast<int64_t>(value));
					}
				} else {
					[]<bool flag = false>() { static_assert(flag, "unsupported member type"); }
					();
				}
				TODO();
			},
			.member_setter = [member, name = std::string(name)](
								 PyObject *self, PyObject *value) -> PyResult<std::monostate> {
				using MemberType = std::remove_pointer_t<
					std::remove_reference_t<decltype(static_cast<T *>(self)->*member)>>;
				if constexpr (std::is_same_v<MemberType, PyObject>) {
					static_cast<T *>(self)->*member = value;
				} else if constexpr (std::is_base_of_v<PyObject, MemberType>) {
					auto *type = (static_cast<T *>(self)->*member)->type();
					if (auto *obj = as<MemberType>(value)) {
						static_cast<T *>(self)->*member = obj;
					} else {
						return Err(type_error("'{}' must be a '{}', not '{}'",
							name,
							type->name(),
							value->type()->name()));
					}
				} else {
					[]<bool flag = false>() { static_assert(flag, "unsupported member type"); }
					();
				}
				return Ok(std::monostate{});
			},
		});
		return *this;
	}

	template<typename ClassMemberType>
	klass &attribute_readonly(std::string_view name, ClassMemberType &&member)
		requires requires(PyObject *self) { static_cast<T *>(self)->*member; }
	{
		type->add_member(MemberDefinition{
			.name = std::string(name),
			.member_accessor = [member](PyObject *self) -> PyResult<PyObject *> {
				static_assert(member_pointer<ClassMemberType>{});
				using MemberType =
					typename std::remove_cvref_t<typename member_pointer<ClassMemberType>::type>;

				if constexpr (std::is_convertible_v<MemberType, PyObject *>) {
					PyObject *obj = static_cast<T *>(self)->*member;
					if (!obj) { return Ok(py_none()); }
					return Ok(obj);
				} else if constexpr (std::numeric_limits<MemberType>::is_integer) {
					if constexpr (std::numeric_limits<MemberType>::is_signed) {
						return PyInteger::create(
							static_cast<int64_t>(static_cast<T *>(self)->*member));
					} else {
						const auto value = static_cast<uint64_t>(static_cast<T *>(self)->*member);
						if (value > std::numeric_limits<int64_t>::max()) { TODO(); }
						return PyInteger::create(static_cast<int64_t>(value));
					}
				} else if constexpr (std::is_same_v<MemberType, std::string>) {
					return PyString::create(static_cast<T *>(self)->*member);
				} else {
					[]<bool flag = false>() { static_assert(flag, "unsupported member type"); }
					();
				}
				TODO();
			},
		});
		return *this;
	}


	template<typename FuncType>
	klass &def(std::string_view name, FuncType &&F)
		requires requires(PyObject *self) { (static_cast<T *>(self)->*F)(); }
	{
		type->add_method(MethodDefinition{
			.name = std::string(name),
			.method = [F, name](
						  PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
				const size_t arg_count = (args ? args->size() : 0) + (kwargs ? kwargs->size() : 0);
				if (arg_count) {
					return Err(
						type_error("{}() takes no arguments ({} given)", name, args->size()));
				}
				if (kwargs && kwargs->size() > 0) {
					return Err(type_error("{}() takes no keyword arguments)", name));
				}
				return (static_cast<T *>(self)->*F)();
			},
			.flags = MethodFlags::create(MethodFlags::Flag::NOARGS),
			.doc = "",
		});
		return *this;
	}

	template<typename FuncType>
	klass &def(std::string_view name, FuncType &&F)
		requires requires(PyObject *self, PyObject *arg0) { (static_cast<T *>(self)->*F)(arg0); }
	{
		std::string name_str{ name };
		type->add_method(MethodDefinition{
			.name = name_str,
			.method = [F, name_str = std::move(name_str)](
						  PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
				auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
					kwargs,
					name_str,
					std::integral_constant<size_t, 1>{},
					std::integral_constant<size_t, 1>{});
				if (result.is_err()) return Err(result.unwrap_err());
				return (static_cast<T *>(self)->*F)(std::get<0>(result.unwrap()));
			},
			.flags = MethodFlags::create(MethodFlags::Flag::O),
			.doc = "",
		});
		return *this;
	}

	template<typename FuncType>
	klass &def(std::string_view name, FuncType &&F)
		requires requires(PyObject *self, PyTuple *args, PyDict *kwargs) {
			(static_cast<T *>(self)->*F)(args, kwargs);
		}
	{
		type->add_method(MethodDefinition{
			.name = std::string(name),
			.method = [F](PyObject *self,
						  PyTuple *args,
						  PyDict *kwargs) { return (static_cast<T *>(self)->*F)(args, kwargs); },
			.flags = MethodFlags::create(),
			.doc = "",
		});
		return *this;
	}

	template<typename FuncType>
	klass &def(std::string_view name, FuncType &&F)
		requires requires(PyObject *self, PyTuple *args, PyDict *kwargs) {
			F(static_cast<T *>(self), args, kwargs);
		}
	{
		type->add_method(MethodDefinition{
			.name = std::string(name),
			.method = [F](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
				return F(static_cast<T *>(self), args, kwargs);
			},
			.flags = MethodFlags::create(),
			.doc = "",
		});
		return *this;
	}

	template<typename FuncType>
	klass &classmethod(std::string_view name, FuncType &&F)
		requires requires(PyType *type, PyTuple *args, PyDict *kwargs) { F(type, args, kwargs); }
	{
		type->add_method(MethodDefinition{
			.name = std::string(name),
			.method = [F](PyObject *type, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
				return F(static_cast<PyType *>(type), args, kwargs);
			},
			.flags = MethodFlags::create(MethodFlags::Flag::CLASSMETHOD),
			.doc = std::string{ "Missing docs" },
		});
		return *this;
	}

	template<typename FuncType>
	klass &staticmethod(std::string_view name, FuncType &&F)
		requires requires(PyTuple *args, PyDict *kwargs) { F(args, kwargs); }
	{
		type->add_method(MethodDefinition{
			.name = std::string(name),
			.method = [F](PyObject *, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
				return F(args, kwargs);
			},
			.flags = MethodFlags::create(MethodFlags::Flag::STATICMETHOD),
			.doc = std::string{ "Missing docs" },
		});
		return *this;
	}

	klass &disable_new()
	{
		type->__new__ = std::nullopt;
		return *this;
	}

	PyType *finalize()
	{
		// FIXME: if we ever add persistent gc pointers this is a place to use them.
		//        klass isn't visited by the GC visitors so `type` is not visited, and
		//        and the allocated `__bases__` tuple can be GC'ed before PyType takes
		//        ownership
		[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
		auto *type_ = PyType::initialize(std::move(type));
		spdlog::trace("Added type@{} with name {}", (void *)type_, type_->name());
		auto name = PyString::create(type_->name());
		if (name.is_err()) { TODO(); }
		m_module->add_symbol(name.unwrap(), type_);
		return type_;
	}
};

}// namespace py
