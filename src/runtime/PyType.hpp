#pragma once

#include "PyObject.hpp"


class PyType : public PyBaseObject
{
	template<typename T> friend struct klass;
	friend class Heap;

  private:
	TypePrototype m_underlying_type;

	PyType(TypePrototype type_prototype);

  public:
	static PyType *initialize(TypePrototype type_prototype);

	const std::string &name() const { return m_underlying_type.__name__; }

	PyObject *__call__(PyTuple *args, PyDict *kwargs) const;
	PyObject *__repr__() const;

	PyObject *new_(PyTuple *args, PyDict *kwargs) const override;

	static PyObject *__new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	std::string to_string() const override;

	const TypePrototype &underlying_type() const { return m_underlying_type; }

	void visit_graph(Visitor &visitor) override { PyObject::visit_graph(visitor); }

	void initialize();

	static std::unique_ptr<TypePrototype> register_type();

	PyType *type() const override;

	PyList *mro();

	bool issubclass(const PyType*);

  private:
	PyTuple *mro_internal();
};
