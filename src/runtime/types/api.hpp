#include "runtime/PyType.hpp"

template<typename T> struct klass
{
	PyType *m_type;

	klass(PyModule *module, std::string_view name)
	{
		auto *name_ = PyString::create(std::string(name));
		m_type = PyType::create<T>(name_);
		module->insert(name_, m_type);
	}

	template<typename FuncType>
	klass &def(std::string_view name, FuncType &&F) requires requires(PyObject *self)
	{
		(static_cast<T *>(self)->*F)();
	}
	{
		m_type->add_method(VirtualMachine::the().heap().allocate<PyMethodDescriptor>(
			PyString::create(std::string(name)), m_type, [F](PyObject *self, PyTuple *args, PyDict *kwargs) {
				// TODO: this should raise an exception
				//       TypeError: {}() takes no arguments ({} given)
				//       TypeError: {}() takes no keyword arguments
				ASSERT(!args)
				ASSERT(!kwargs)
				return (static_cast<T *>(self)->*F)();
			}));
		return *this;
	}
};