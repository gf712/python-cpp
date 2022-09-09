#pragma once

#include "builtin.hpp"
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
	requires(std::is_same_v<std::remove_reference_t<BaseType>, PyType *> &&...)
		klass(PyModule *module, std::string_view name, BaseType &&...bases)
		: type(TypePrototype::create<T>(name)), m_module(module)
	{
		type->__bases__ = std::vector<PyObject *>{ bases... };
	}

	klass(std::string_view name) : type(TypePrototype::create<T>(name)) {}

	template<typename... BaseType>
	requires(std::is_same_v<std::remove_reference_t<BaseType>, PyType *> &&...)
		klass(std::string_view name, BaseType &&...bases)
		: type(TypePrototype::create<T>(name))
	{
		type->__bases__ = std::vector<PyObject *>{ bases... };
	}

	using AttributeGetterType = std::function<PyObject *(T *)>;
	using AttributeSetterType = std::function<PyResult<std::monostate>(T *, PyObject *)>;

	klass &property(std::string_view name,
		std::optional<AttributeGetterType> &&getter,
		std::optional<AttributeSetterType> &&setter)
	{
		type->add_property(PropertyDefinition{
			.name = std::string(name),
			.member_getter = [&getter]() -> std::optional<std::function<PyObject *(PyObject *)>> {
				if (getter.has_value()) {
					return [getter_ = std::move(getter)](PyObject *self) -> PyObject * {
						return getter_->operator()(static_cast<T *>(self));
					};
				} else {
					return std::nullopt;
				}
			}(),
			.member_setter = [&setter]()
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
	klass &attr(std::string_view name, ClassMemberType &&member) requires requires(PyObject *self)
	{
		static_cast<T *>(self)->*member;
	}
	{
		type->add_member(MemberDefinition{
			.name = std::string(name),
			.member_accessor = [member](PyObject *self) -> PyObject * {
				static_assert(member_pointer<ClassMemberType>{});
				using MemberType =
					typename std::remove_cvref_t<typename member_pointer<ClassMemberType>::type>;

				if constexpr (std::is_convertible_v<MemberType, PyObject *>) {
					PyObject *obj = static_cast<T *>(self)->*member;
					if (!obj) { return py_none(); }
					return obj;
				} else if constexpr (std::numeric_limits<MemberType>::is_integer) {
					if constexpr (std::numeric_limits<MemberType>::is_signed) {
						return PyInteger::create(
							static_cast<int64_t>(static_cast<T *>(self)->*member))
							.unwrap();
					} else {
						const auto value = static_cast<uint64_t>(static_cast<T *>(self)->*member);
						if (value > std::numeric_limits<int64_t>::max()) { TODO(); }
						return PyInteger::create(static_cast<int64_t>(value)).unwrap();
					}
				} else {
					[]<bool flag = false>() { static_assert(flag, "unsupported member type"); }
					();
				}
			},
			.member_setter = [member](PyObject *self, PyObject *value) -> PyResult<std::monostate> {
				using MemberType = std::remove_pointer_t<
					std::remove_reference_t<decltype(static_cast<T *>(self)->*member)>>;
				if constexpr (std::is_same_v<MemberType, PyObject>) {
					static_cast<T *>(self)->*member = value;
				} else if constexpr (std::is_base_of_v<PyObject, MemberType>) {
					auto *type = (static_cast<T *>(self)->*member)->type();
					if (auto *obj = as<MemberType>(value)) {
						static_cast<T *>(self)->*member = obj;
					} else {
						return Err(type_error("attribute value type must be '{}'", type->name()));
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
	klass &attribute_readonly(std::string_view name, ClassMemberType &&member) requires
		requires(PyObject *self)
	{
		static_cast<T *>(self)->*member;
	}
	{
		type->add_member(MemberDefinition{
			.name = std::string(name),
			.member_accessor = [member](PyObject *self) -> PyObject * {
				static_assert(member_pointer<ClassMemberType>{});
				using MemberType =
					typename std::remove_cvref_t<typename member_pointer<ClassMemberType>::type>;

				if constexpr (std::is_convertible_v<MemberType, PyObject *>) {
					PyObject *obj = static_cast<T *>(self)->*member;
					if (!obj) { return py_none(); }
					return obj;
				} else if constexpr (std::numeric_limits<MemberType>::is_integer) {
					if constexpr (std::numeric_limits<MemberType>::is_signed) {
						return PyInteger::create(
							static_cast<int64_t>(static_cast<T *>(self)->*member))
							.unwrap();
					} else {
						const auto value = static_cast<uint64_t>(static_cast<T *>(self)->*member);
						if (value > std::numeric_limits<int64_t>::max()) { TODO(); }
						return PyInteger::create(static_cast<int64_t>(value)).unwrap();
					}
				} else if constexpr (std::is_same_v<MemberType, std::string>) {
					return PyString::create(static_cast<T *>(self)->*member).unwrap();
				} else {
					[]<bool flag = false>() { static_assert(flag, "unsupported member type"); }
					();
				}
			},
		});
		return *this;
	}


	template<typename FuncType>
	klass &def(std::string_view name, FuncType &&F) requires requires(PyObject *self)
	{
		(static_cast<T *>(self)->*F)();
	}
	{
		type->add_method(MethodDefinition{
			.name = std::string(name),
			.method =
				[F](PyObject *self, PyTuple *args, PyDict *kwargs) {
					// TODO: this should raise an exception
					//       TypeError: {}() takes no arguments ({} given)
					//       TypeError: {}() takes no keyword arguments
					ASSERT(!args || args->size() == 0)
					ASSERT(!kwargs || kwargs->map().empty())
					return (static_cast<T *>(self)->*F)();
				},
			.flags = MethodFlags::create(),
			.doc = "",
		});
		return *this;
	}

	template<typename FuncType>
	klass &def(std::string_view name, FuncType &&F) requires
		requires(PyObject *self, PyTuple *args, PyDict *kwargs)
	{
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
	klass &def(std::string_view name, FuncType &&F) requires
		requires(PyObject *self, PyTuple *args, PyDict *kwargs)
	{
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
	klass &classmethod(std::string_view name, FuncType &&F) requires
		requires(PyType *type, PyTuple *args, PyDict *kwargs)
	{
		F(type, args, kwargs);
	}
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
	klass &staticmethod(std::string_view name, FuncType &&F) requires
		requires(PyTuple *args, PyDict *kwargs)
	{
		F(args, kwargs);
	}
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
		auto *type_ = PyType::initialize(std::move(type));
		spdlog::trace("Added type@{} with name {}", (void *)type_, type_->name());
		auto name = PyString::create(type_->name());
		if (name.is_err()) { TODO(); }
		m_module->add_symbol(name.unwrap(), type_);
		return type_;
	}
};

}// namespace py