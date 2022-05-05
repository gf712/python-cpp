#pragma once

#include "builtin.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"

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
		klass(PyModule *module, std::string_view name, BaseType &&... bases)
		: type(TypePrototype::create<T>(name)), m_module(module)
	{
		auto bases_ = PyTuple::create(bases...);
		if (bases_.is_err()) { TODO(); }
		type->__bases__ = bases_.unwrap();
	}

	klass(std::string_view name) : type(TypePrototype::create<T>(name)) {}

	template<typename MemberType>
	klass &attr(std::string_view name, MemberType &&member) requires requires(PyObject *self)
	{
		static_cast<T *>(self)->*member;
	}
	{
		type->add_member(MemberDefinition{
			.name = std::string(name), .member_accessor = [member](PyObject *self) -> PyObject * {
				if (!member) { return py_none(); }
				return static_cast<T *>(self)->*member;
			} });
		return *this;
	}

	template<typename FuncType>
	klass &def(std::string_view name, FuncType &&F) requires requires(PyObject *self)
	{
		(static_cast<T *>(self)->*F)();
	}
	{
		type->add_method(MethodDefinition{
			std::string(name), [F](PyObject *self, PyTuple *args, PyDict *kwargs) {
				// TODO: this should raise an exception
				//       TypeError: {}() takes no arguments ({} given)
				//       TypeError: {}() takes no keyword arguments
				ASSERT(!args || args->size() == 0)
				ASSERT(!kwargs || kwargs->map().empty())
				return (static_cast<T *>(self)->*F)();
			} });
		return *this;
	}

	template<typename FuncType>
	klass &def(std::string_view name,
		FuncType &&F) requires requires(PyObject *self, PyTuple *args, PyDict *kwargs)
	{
		(static_cast<T *>(self)->*F)(args, kwargs);
	}
	{
		type->add_method(MethodDefinition{
			std::string(name), [F](PyObject *self, PyTuple *args, PyDict *kwargs) {
				return (static_cast<T *>(self)->*F)(args, kwargs);
			} });
		return *this;
	}

	template<typename FuncType>
	klass &def(std::string_view name,
		FuncType &&F) requires requires(PyObject *self, PyTuple *args, PyDict *kwargs)
	{
		F(static_cast<T *>(self), args, kwargs);
	}
	{
		type->add_method(MethodDefinition{ std::string(name),
			[F](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
				return F(static_cast<T *>(self), args, kwargs);
			} });
		return *this;
	}

	PyType *finalize()
	{
		auto *type_ = PyType::initialize(*type.release());
		auto name = PyString::create(type_->name());
		if (name.is_err()) { TODO(); }
		m_module->insert(name.unwrap(), type_);
		return type_;
	}
};

}// namespace py