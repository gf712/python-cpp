#include "CustomPyObject.hpp"
#include "PyBoundMethod.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"


CustomPyObject::CustomPyObject(const PyType *type)
	: PyBaseObject(PyObjectType::PY_CUSTOM_TYPE, type->underlying_type()), m_type_obj(type)
{
}

PyObject *CustomPyObject::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	if (args && !args->elements().empty()) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("object() takes no arguments"));
		return nullptr;
	}
	if (kwargs && !kwargs->map().empty()) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("object() takes no arguments"));
		return nullptr;
	}
	return VirtualMachine::the().heap().allocate<CustomPyObject>(type);
}

PyType *CustomPyObject::type_() const
{
	// FIXME: should type_ return const PyType* instead?
	return const_cast<PyType *>(m_type_obj);
}

namespace {

std::once_flag object_flag;

std::unique_ptr<TypePrototype> register_object()
{
	return std::move(klass<CustomPyObject>("object").type);
}
}// namespace

std::unique_ptr<TypePrototype> CustomPyObject::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(object_flag, []() { type = ::register_object(); });
	return std::move(type);
}