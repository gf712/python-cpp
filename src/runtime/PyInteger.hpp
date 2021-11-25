#include "PyNumber.hpp"

class PyInteger final : public PyNumber
{
	friend class Heap;

	PyInteger(int64_t);

  public:
	static PyInteger *create(int64_t);

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type_() const override;
};


template<> inline PyInteger *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_INTEGER) { return static_cast<PyInteger *>(node); }
	return nullptr;
}

template<> inline const PyInteger *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_INTEGER) { return static_cast<const PyInteger *>(node); }
	return nullptr;
}
