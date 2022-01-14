#include "PyBool.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

PyBool::PyBool(bool name) : PyBaseObject(BuiltinTypes::the().bool_()), m_value(name) {}

std::string PyBool::to_string() const { return m_value ? "True" : "False"; }

PyObject *PyBool::__repr__() const { return PyString::from(String{ to_string() }); }

PyObject *PyBool::__add__(const PyObject *) const { TODO(); }

PyObject *PyBool::__bool__() const { return m_value ? py_true() : py_false(); }

PyBool *PyBool::create(bool value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate_static<PyBool>(value).get();
}

PyType *PyBool::type() const { return ::bool_(); }

PyObject *py::py_true()
{
	static PyObject *value = nullptr;

	if (!value) { value = PyBool::create(true); }

	return value;
}

PyObject *py::py_false()
{
	static PyObject *value = nullptr;

	if (!value) { value = PyBool::create(false); }

	return value;
}


namespace {

std::once_flag bool_flag;

std::unique_ptr<TypePrototype> register_bool() { return std::move(klass<PyBool>("bool").type); }
}// namespace

std::unique_ptr<TypePrototype> PyBool::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(bool_flag, []() { type = ::register_bool(); });
	return std::move(type);
}

template<> PyBool *py::as(PyObject *node)
{
	if (node->type() == bool_()) { return static_cast<PyBool *>(node); }
	return nullptr;
}

template<> const PyBool *py::as(const PyObject *node)
{
	if (node->type() == bool_()) { return static_cast<const PyBool *>(node); }
	return nullptr;
}