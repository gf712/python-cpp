#include "PyNumber.hpp"

namespace py {

class PyFloat final : public PyNumber
{
	friend class ::Heap;

	PyFloat(double);

  public:
	static PyFloat *create(double);
	PyType *type() const override;

	double as_f64() const;

	static std::unique_ptr<TypePrototype> register_type();
};

}// namespace py