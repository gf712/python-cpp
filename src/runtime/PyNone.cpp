#include "PyNone.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

PyNone::PyNone() : PyBaseObject(BuiltinTypes::the().none()) {}

std::string PyNone::to_string() const { return "None"; }

PyResult PyNone::__repr__() const { return PyString::create(to_string()); }

PyResult PyNone::__add__(const PyObject *) const { TODO(); }

PyNone *PyNone::create()
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate_static<PyNone>().get();
}

PyType *PyNone::type() const { return none(); }

namespace {

std::once_flag none_flag;

std::unique_ptr<TypePrototype> register_none() { return std::move(klass<PyNone>("NoneType").type); }
}// namespace

std::unique_ptr<TypePrototype> PyNone::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(none_flag, []() { type = ::register_none(); });
	return std::move(type);
}

PyObject *py::py_none()
{
	static PyObject *value = nullptr;

	if (!value) { value = PyNone::create(); }

	return value;
}
