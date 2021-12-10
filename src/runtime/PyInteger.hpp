#include "PyNumber.hpp"

class PyInteger final : public PyNumber
{
	friend class Heap;

	PyInteger(int64_t);

  public:
	static PyInteger *create(int64_t);

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};
