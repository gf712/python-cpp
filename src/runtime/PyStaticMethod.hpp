#pragma once

#include "PyObject.hpp"

class PyStaticMethod : public PyBaseObject
{
	PyString *m_name;
	PyType *m_underlying_type;
	std::function<PyObject *(PyType *, PyTuple *, PyDict *)> m_static_method;

	friend class Heap;

	PyStaticMethod(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyType *, PyTuple *, PyDict *)> function);

  public:
	static PyStaticMethod *create(PyString *name,
		PyType *underlying_type,
		std::function<PyObject *(PyType *, PyTuple *, PyDict *)> function);

	PyString *static_method_name() { return m_name; }
	const std::function<PyObject *(PyType *, PyTuple *, PyDict *)> &static_method() { return m_static_method; }

	std::string to_string() const override;

	PyObject *__repr__() const;
	PyObject *__call__(PyTuple *args, PyDict *kwargs);

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