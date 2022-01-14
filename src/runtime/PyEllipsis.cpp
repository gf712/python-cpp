#include "PyEllipsis.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

PyEllipsis::PyEllipsis() : PyBaseObject(BuiltinTypes::the().ellipsis()) {}

PyEllipsis *PyEllipsis::create()
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate_static<PyEllipsis>().get();
}

PyObject *PyEllipsis::__add__(const PyObject *) const { TODO(); }

PyObject *PyEllipsis::__repr__() const { return PyString::create("Ellipsis"); }

PyType *PyEllipsis::type() const { return ellipsis(); }

PyObject *py::py_ellipsis()
{
	static PyObject *ellipsis = nullptr;
	if (!ellipsis) { ellipsis = PyEllipsis::create(); }
	return ellipsis;
}

namespace {

std::once_flag ellipsis_flag;

std::unique_ptr<TypePrototype> register_ellipsis()
{
	return std::move(klass<PyEllipsis>("ellipsis").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyEllipsis::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(ellipsis_flag, []() { type = ::register_ellipsis(); });
	return std::move(type);
}
