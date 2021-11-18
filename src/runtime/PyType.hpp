#include "PyDict.hpp"
#include "PyModule.hpp"
#include "PyObject.hpp"
#include "PyString.hpp"
#include "vm/VM.hpp"

class PyBoundMethod : public PyBaseObject<PyBoundMethod>
{
	PyObject *m_self;
	PyFunction *m_method;

  public:
	PyBoundMethod(PyObject *self, PyFunction *method)
		: PyBaseObject(PyObjectType::PY_BOUND_METHOD), m_self(self), m_method(method)
	{}

	PyObject *self() { return m_self; }
	PyFunction *method() { return m_method; }

	std::string to_string() const override;

	PyObject *repr_impl() const { return PyString::create(to_string()); }

	void visit_graph(Visitor &visitor) override;
};

class PyMethodDescriptor : public PyBaseObject<PyMethodDescriptor>
{
	PyString *m_name;
	PyType *m_underlying_type;
	std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> m_method_descriptor;
	std::vector<PyObject *> m_captures;

  public:
	template<typename... Args>
	PyMethodDescriptor(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function,
		Args &&... args)
		: PyBaseObject(PyObjectType::PY_METHOD_DESCRIPTOR), m_name(std::move(name)),
		  m_underlying_type(underlying_type),
		  m_method_descriptor(std::move(function)), m_captures{ std::forward<Args>(args)... }
	{}

	PyString *name() { return m_name; }
	const std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> &method_descriptor()
	{
		return m_method_descriptor;
	}

	std::string to_string() const override;

	PyObject *repr_impl() const { return PyString::create(to_string()); }

	void visit_graph(Visitor &visitor) override;
};

class PySlotWrapper : public PyBaseObject<PySlotWrapper>
{
	PyString *m_name;
	PyType *m_underlying_type;
	std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> m_slot;

  public:
	template<typename... Args>
	PySlotWrapper(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function)
		: PyBaseObject(PyObjectType::PY_SLOT_WRAPPER), m_name(std::move(name)),
		  m_underlying_type(underlying_type), m_slot(std::move(function))
	{}

	PyString *name() { return m_name; }
	const std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> &slot() { return m_slot; }

	std::string to_string() const override;

	PyObject *repr_impl() const { return PyString::create(to_string()); }

	void visit_graph(Visitor &visitor) override;
};

class PyType : public PyBaseObject<PyType>
{
	template<typename T> friend struct klass;
	friend class Heap;

	struct SlotWrappers
	{
		PySlotWrapper *__repr__{ nullptr };
	};
	SlotWrappers m_slot_wrappers;

	std::vector<PyMethodDescriptor *> __methods__;

	// mappingproxy
	PyDict *__dict__{ nullptr };
#

	PyType(PyString *type_name) : PyBaseObject(PyObjectType::PY_TYPE), m_name(type_name) {}

	PyString *m_name;

  public:
	template<typename Type> static PyType *create(PyString *type_name)
	{
		auto *type = VirtualMachine::the().heap().allocate<PyType>(type_name);
		type->__dict__ = VirtualMachine::the().heap().allocate<PyDict>();
		if constexpr (HasRepr<Type>) {
			type->m_slot_wrappers.__repr__ =
				VirtualMachine::the().heap().allocate<PySlotWrapper>(PyString::create("__repr__"),
					type,
					[](PyObject *obj, PyTuple *args, PyDict *kwargs) {
						// TODO: this should raise an exception
						//       TypeError: {}() takes no arguments ({} given)
						//       TypeError: {}() takes no keyword arguments
						ASSERT(!args)
						ASSERT(!kwargs)
						return obj->__repr__();
					});
			type->__dict__->insert(PyString::create("__repr__"), type->m_slot_wrappers.__repr__);
		}

		type->m_attributes["__dict__"] = type->__dict__;

		return type;
	}

	PyString *name() const { return m_name; }

	PyObject *repr_impl() const { return PyString::create(to_string()); }

	std::string to_string() const override { return fmt::format("class <{}>", m_name->value()); }

	void add_method(PyMethodDescriptor *method)
	{
		__methods__.push_back(method);
		__dict__->insert(method->name(), method);
		m_attributes[method->name()->value()] = method;
	}

	void visit_graph(Visitor &visitor) override
	{
		PyObject::visit_graph(visitor);
		visitor.visit(*m_name);
		if (m_slot_wrappers.__repr__) { visitor.visit(*m_slot_wrappers.__repr__); }
		for (auto *method : __methods__) { visitor.visit(*method); }
		if (__dict__) { visitor.visit(*__dict__); }
	}
};


template<> inline PyBoundMethod *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_BOUND_METHOD) { return static_cast<PyBoundMethod *>(node); }
	return nullptr;
}

template<> inline const PyBoundMethod *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_BOUND_METHOD) { return static_cast<const PyBoundMethod *>(node); }
	return nullptr;
}


template<> inline PyMethodDescriptor *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_METHOD_DESCRIPTOR) {
		return static_cast<PyMethodDescriptor *>(node);
	}
	return nullptr;
}

template<> inline const PyMethodDescriptor *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_METHOD_DESCRIPTOR) {
		return static_cast<const PyMethodDescriptor *>(node);
	}
	return nullptr;
}

template<> inline PySlotWrapper *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_SLOT_WRAPPER) {
		return static_cast<PySlotWrapper *>(node);
	}
	return nullptr;
}

template<> inline const PySlotWrapper *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_SLOT_WRAPPER) {
		return static_cast<const PySlotWrapper *>(node);
	}
	return nullptr;
}

template<> inline PyType *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_TYPE) { return static_cast<PyType *>(node); }
	return nullptr;
}

template<> inline const PyType *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_TYPE) { return static_cast<const PyType *>(node); }
	return nullptr;
}
