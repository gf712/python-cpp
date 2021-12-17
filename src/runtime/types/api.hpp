#pragma once

#include "builtin.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"

template<typename T> struct klass
{
	std::unique_ptr<TypePrototype> type;

	klass(PyModule *module, std::string_view name)
	{
		(void)module;
		(void)name;
		TODO();
		// module->insert(name_, type);
	}

	klass(std::string_view name) : type(TypePrototype::create<T>(name)) {}

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
};