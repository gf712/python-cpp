#pragma once

#include "PyObject.hpp"

namespace py {

class PyType : public PyBaseObject
{
	template<typename T> friend struct klass;
	friend class ::Heap;
	friend class PyObject;

  private:
	// FIXME: this is mutable to enable some caching behaviour
	// 		  for example MRO is only computed the first time it is requested
	mutable TypePrototype m_underlying_type;

	PyType(TypePrototype type_prototype);

  public:
	static PyType *initialize(TypePrototype type_prototype);

	const std::string &name() const { return m_underlying_type.__name__; }

	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __getattribute__(PyObject *attribute) const;

	PyResult<PyObject *> new_(PyTuple *args, PyDict *kwargs) const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	std::string to_string() const override;

	const TypePrototype &underlying_type() const { return m_underlying_type; }

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();

	PyType *type() const override;

	PyResult<PyList *> mro();

	bool issubclass(const PyType *);

	PyResult<PyObject *> lookup(PyObject *name) const;

  protected:
	PyResult<PyTuple *> mro_internal() const;

  private:
	void initialize(PyDict *ns);
	void update_methods_and_class_attributes(PyDict *ns);
	bool update_if_special(const std::string &name, const Value &value);

	static PyResult<PyType *> build_type(PyString *type_name, PyTuple *bases, PyDict *ns);
};

}// namespace py