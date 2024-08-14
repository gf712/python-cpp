#include "PyComplex.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyType.hpp"
#include "runtime/types/api.hpp"

using namespace py;

PyComplex::PyComplex(PyType *type) : PyBaseObject(type) {}

PyComplex::PyComplex(TypePrototype &type, std::complex<BigIntType> complex)
	: PyBaseObject(type), m_complex(std::move(complex))
{}

PyComplex::PyComplex(PyType *type, std::complex<BigIntType> complex)
	: PyBaseObject(type), m_complex(std::move(complex))
{}

PyType *PyComplex::static_type() const { return types::complex(); }

namespace {

std::once_flag complex_flag;

std::unique_ptr<TypePrototype> register_complex()
{
	return std::move(klass<PyComplex>("complex").type);
}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyComplex::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(complex_flag, []() { type = register_complex(); });
		return std::move(type);
	};
}
