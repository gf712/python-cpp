#include "PyNumber.hpp"

namespace py {

class PyInteger final : public PyNumber
{
	friend class ::Heap;

	PyInteger(int64_t);

  public:
	static PyInteger *create(int64_t);

	static PyInteger *__new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
	size_t as_size_t() const;
};

}// namespace py