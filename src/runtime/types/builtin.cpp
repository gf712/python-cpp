#include "builtin.hpp"

#include "runtime/PyBool.hpp"
#include "runtime/PyBoundMethod.hpp"
#include "runtime/PyBuiltInMethod.hpp"
#include "runtime/PyBytes.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyEllipsis.hpp"
#include "runtime/PyFloat.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyMethodDescriptor.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyProperty.hpp"
#include "runtime/PyRange.hpp"
#include "runtime/PySlotWrapper.hpp"
#include "runtime/PyStaticMethod.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"

using namespace py;

BuiltinTypes::BuiltinTypes()
	: m_type(PyType::register_type()), m_str(PyString::register_type()),
	  m_bool(PyBool::register_type()), m_bytes(PyBytes::register_type()),
	  m_ellipsis(PyEllipsis::register_type()), m_float(PyFloat::register_type()),
	  m_integer(PyInteger::register_type()), m_none(PyNone::register_type()),
	  m_module(PyModule::register_type()), m_object(PyObject::register_type()),
	  m_function(PyFunction::register_type()), m_native_function(PyNativeFunction::register_type()),
	  m_code(PyCode::register_type()), m_dict(PyDict::register_type()),
	  m_dict_items(PyDictItems::register_type()),
	  m_dict_items_iterator(PyDictItemsIterator::register_type()), m_list(PyList::register_type()),
	  m_list_iterator(PyListIterator::register_type()), m_tuple(PyTuple::register_type()),
	  m_tuple_iterator(PyTupleIterator::register_type()), m_range(PyRange::register_type()),
	  m_range_iterator(PyRangeIterator::register_type()),
	  m_builtin_method(PyBuiltInMethod::register_type()),
	  m_slot_wrapper(PySlotWrapper::register_type()),
	  m_bound_method(PyBoundMethod::register_type()),
	  m_method_wrapper(PyMethodDescriptor::register_type()),
	  m_static_method(PyStaticMethod::register_type()), m_property(PyProperty::register_type())
{}

#define INITIALIZE_TYPE(TYPENAME)                             \
	PyType *TYPENAME()                                        \
	{                                                         \
		static PyType *type = nullptr;                        \
		if (!type) {                                          \
			auto &prototype = BuiltinTypes::the().TYPENAME(); \
			type = PyType::initialize(prototype);             \
		}                                                     \
		return type;                                          \
	}

namespace py {
INITIALIZE_TYPE(type)
INITIALIZE_TYPE(bool_)
INITIALIZE_TYPE(bytes)
INITIALIZE_TYPE(ellipsis)
INITIALIZE_TYPE(str)
INITIALIZE_TYPE(float_)
INITIALIZE_TYPE(integer)
INITIALIZE_TYPE(none)
INITIALIZE_TYPE(module)
INITIALIZE_TYPE(object)

INITIALIZE_TYPE(dict)
INITIALIZE_TYPE(dict_items)
INITIALIZE_TYPE(dict_items_iterator)

INITIALIZE_TYPE(list)
INITIALIZE_TYPE(list_iterator)

INITIALIZE_TYPE(tuple)
INITIALIZE_TYPE(tuple_iterator)

INITIALIZE_TYPE(range)
INITIALIZE_TYPE(range_iterator)

INITIALIZE_TYPE(function)
INITIALIZE_TYPE(native_function)
INITIALIZE_TYPE(code)

INITIALIZE_TYPE(builtin_method)
INITIALIZE_TYPE(slot_wrapper)
INITIALIZE_TYPE(bound_method)
INITIALIZE_TYPE(method_wrapper)
INITIALIZE_TYPE(static_method)
INITIALIZE_TYPE(property)
}// namespace py