#include "PyNumber.hpp"

class PyFloat final: public PyNumber
{
	friend class Heap;

	PyFloat(double);

  public:
	static PyFloat *create(double);
	PyType *type_() const override;

	static std::unique_ptr<TypePrototype> register_type();
};


template<> inline PyFloat *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_FLOAT) { return static_cast<PyFloat *>(node); }
	return nullptr;
}

template<> inline const PyFloat *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_FLOAT) { return static_cast<const PyFloat *>(node); }
	return nullptr;
}
