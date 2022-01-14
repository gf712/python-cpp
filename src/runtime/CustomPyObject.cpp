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

using namespace py;

CustomPyObject::CustomPyObject(const PyType *type)
	: PyBaseObject(type->underlying_type()), m_type_obj(type)
{}

PyType *CustomPyObject::type() const
{
	// FIXME: should type_ return const PyType* instead?
	return const_cast<PyType *>(m_type_obj);
}

std::string CustomPyObject::to_string() const {
	return fmt::format("CustomPyObject of type \"{}\"", type()->to_string());
}
