#include "PyNumber.hpp"

class PyFloat final: public PyNumber
{
	friend class Heap;

	PyFloat(double);

  public:
	static PyFloat *create(double);
	PyType *type() const override;

	static std::unique_ptr<TypePrototype> register_type();
};
