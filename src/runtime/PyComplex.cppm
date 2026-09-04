module;
#include "memory/allocate.hpp"

export module py.runtime:complex;
import :object;
import py.memory;
import std;

export namespace py {
class PyType;

class PyComplex : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	std::complex<BigIntType> m_complex;

  protected:
	PyComplex(PyType *);

	PyComplex(TypePrototype &, std::complex<BigIntType>);

	PyComplex(PyType *, std::complex<BigIntType>);

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
