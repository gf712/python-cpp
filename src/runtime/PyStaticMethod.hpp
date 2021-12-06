#pragma once

#include "PyObject.hpp"

class PyStaticMethod : public PyBaseObject
{
	using TypeBoundFunctionType = std::function<PyObject *(PyType *, PyTuple *, PyDict *)>;
	using FreeFunctionType = std::function<PyObject *(PyTuple *, PyDict *)>;

	PyString *m_name;
	PyType *m_underlying_type{ nullptr };
	std::variant<TypeBoundFunctionType, FreeFunctionType> m_static_method;

	friend class Heap;

	PyStaticMethod(PyString *name,
		PyType *underlying_type,
		std::variant<TypeBoundFunctionType, FreeFunctionType> function);

  public:
	static PyStaticMethod *
		create(PyString *name, PyType *underlying_type, TypeBoundFunctionType function);

	static PyStaticMethod *create(PyString *name, FreeFunctionType function);

	PyString *static_method_name() { return m_name; }
	const std::variant<TypeBoundFunctionType, FreeFunctionType> &static_method()
	{
		return m_static_method;
	}

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *call_static_method(PyTuple *args, PyDict *kwargs);

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type_() const override;
};

template<> inline PyStaticMethod *as(PyObject *obj)
{
	if (obj->type() == PyObjectType::PY_STATIC_METHOD) {
		return static_cast<PyStaticMethod *>(obj);
	}
	return nullptr;
}

template<> inline const PyStaticMethod *as(const PyObject *obj)
{
	if (obj->type() == PyObjectType::PY_STATIC_METHOD) {
		return static_cast<const PyStaticMethod *>(obj);
	}
	return nullptr;
}