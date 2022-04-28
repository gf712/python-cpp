#include "PyEllipsis.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

PyEllipsis::PyEllipsis() : PyBaseObject(BuiltinTypes::the().ellipsis()) {}

PyResult PyEllipsis::create()
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate_static<PyEllipsis>().get();
	if (!obj) { return PyResult::Err(memory_error(sizeof(PyEllipsis))); }
	return PyResult::Ok(obj);
}

PyResult PyEllipsis::__add__(const PyObject *) const { TODO(); }

PyResult PyEllipsis::__repr__() const { return PyString::create("Ellipsis"); }

PyType *PyEllipsis::type() const { return ellipsis(); }

PyObject *py::py_ellipsis()
{
	static PyObject *ellipsis = nullptr;
	if (!ellipsis) {
		auto obj = PyEllipsis::create();
		ASSERT(obj.is_ok())
		ellipsis = obj.unwrap_as<PyObject>();
	}
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
