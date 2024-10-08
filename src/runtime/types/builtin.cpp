#include "builtin.hpp"

#include "runtime/AssertionError.hpp"
#include "runtime/AttributeError.hpp"
#include "runtime/ImportError.hpp"
#include "runtime/IndexError.hpp"
#include "runtime/KeyError.hpp"
#include "runtime/LookupError.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/ModuleNotFoundError.hpp"
#include "runtime/NameError.hpp"
#include "runtime/NotImplemented.hpp"
#include "runtime/NotImplementedError.hpp"
#include "runtime/OSError.hpp"
#include "runtime/PyAsyncGenerator.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyBoundMethod.hpp"
#include "runtime/PyBuiltInMethod.hpp"
#include "runtime/PyByteArray.hpp"
#include "runtime/PyBytes.hpp"
#include "runtime/PyCell.hpp"
#include "runtime/PyClassMethod.hpp"
#include "runtime/PyClassMethodDescriptor.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyComplex.hpp"
#include "runtime/PyCoroutine.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyEllipsis.hpp"
#include "runtime/PyEnumerate.hpp"
#include "runtime/PyFloat.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyFrozenSet.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyGenerator.hpp"
#include "runtime/PyGenericAlias.hpp"
#include "runtime/PyGetSetDescriptor.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyIterator.hpp"
#include "runtime/PyLLVMFunction.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyMap.hpp"
#include "runtime/PyMappingProxy.hpp"
#include "runtime/PyMemberDescriptor.hpp"
#include "runtime/PyMemoryView.hpp"
#include "runtime/PyMethodDescriptor.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyNamespace.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyProperty.hpp"
#include "runtime/PyRange.hpp"
#include "runtime/PyReversed.hpp"
#include "runtime/PySet.hpp"
#include "runtime/PySlice.hpp"
#include "runtime/PySlotWrapper.hpp"
#include "runtime/PyStaticMethod.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PySuper.hpp"
#include "runtime/PyTraceback.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "runtime/PyZip.hpp"
#include "runtime/RuntimeError.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/SyntaxError.hpp"
#include "runtime/UnboundLocalError.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/warnings/DeprecationWarning.hpp"
#include "runtime/warnings/ImportWarning.hpp"
#include "runtime/warnings/PendingDeprecationWarning.hpp"
#include "runtime/warnings/ResourceWarning.hpp"
#include "runtime/warnings/Warning.hpp"

namespace py::types {

BuiltinTypes::BuiltinTypes()
	: m_type(PyType::type_factory()), m_super(PySuper::type_factory()),
	  m_str(PyString::type_factory()), m_str_iterator(PyStringIterator::type_factory()),
	  m_integer(PyInteger::type_factory()), m_complex(PyComplex::type_factory()),
	  m_bool(PyBool::type_factory()), m_bytes(PyBytes::type_factory()),
	  m_bytes_iterator(PyBytesIterator::type_factory()), m_bytearray(PyByteArray::type_factory()),
	  m_bytearray_iterator(PyByteArrayIterator::type_factory()),
	  m_ellipsis(PyEllipsis::type_factory()), m_float(PyFloat::type_factory()),
	  m_none(PyNone::type_factory()), m_module(PyModule::type_factory()),
	  m_object(PyObject::type_factory()), m_memoryview(PyMemoryView::type_factory()),
	  m_function(PyFunction::type_factory()), m_native_function(PyNativeFunction::type_factory()),
	  m_llvm_function(PyLLVMFunction::type_factory()), m_code(PyCode::type_factory()),
	  m_cell(PyCell::type_factory()), m_dict(PyDict::type_factory()),
	  m_dict_items(PyDictItems::type_factory()),
	  m_dict_items_iterator(PyDictItemsIterator::type_factory()),
	  m_dict_keys(PyDictKeys::type_factory()),
	  m_dict_key_iterator(PyDictKeyIterator::type_factory()),
	  m_dict_values(PyDictValues::type_factory()),
	  m_dict_value_iterator(PyDictValueIterator::type_factory()),
	  m_mappingproxy(PyMappingProxy::type_factory()), m_list(PyList::type_factory()),
	  m_list_iterator(PyListIterator::type_factory()),
	  m_list_reverseiterator(PyListReverseIterator::type_factory()),
	  m_tuple(PyTuple::type_factory()), m_tuple_iterator(PyTupleIterator::type_factory()),
	  m_set(PySet::type_factory()), m_frozenset(PyFrozenSet::type_factory()),
	  m_set_iterator(PySetIterator::type_factory()), m_range(PyRange::type_factory()),
	  m_range_iterator(PyRangeIterator::type_factory()), m_reversed(PyReversed::type_factory()),
	  m_zip(PyZip::type_factory()), m_map(PyMap::type_factory()),
	  m_enumerate(PyEnumerate::type_factory()), m_slice(PySlice::type_factory()),
	  m_iterator(PyIterator::type_factory()), m_builtin_method(PyBuiltInMethod::type_factory()),
	  m_slot_wrapper(PySlotWrapper::type_factory()), m_bound_method(PyBoundMethod::type_factory()),
	  m_method_wrapper(PyMethodDescriptor::type_factory()),
	  m_classmethod_descriptor(PyClassMethodDescriptor::type_factory()),
	  m_getset_descriptor(PyGetSetDescriptor::type_factory()),
	  m_static_method(PyStaticMethod::type_factory()), m_property(PyProperty::type_factory()),
	  m_classmethod(PyClassMethod::type_factory()),
	  m_member_descriptor(PyMemberDescriptor::type_factory()),
	  m_traceback(PyTraceback::type_factory()), m_frame(PyFrame::type_factory()),
	  m_not_implemented(NotImplemented::type_factory()), m_namespace(PyNamespace::type_factory()),
	  m_generator(PyGenerator::type_factory()), m_coroutine(PyCoroutine::type_factory()),
	  m_async_generator(PyAsyncGenerator::type_factory()),
	  m_generic_alias(PyGenericAlias::type_factory()),
	  m_base_exception(BaseException::type_factory()), m_exception(Exception::type_factory()),
	  m_type_error(TypeError::type_factory()), m_assertion_error(AssertionError::type_factory()),
	  m_attribute_error(AttributeError::type_factory()), m_value_error(ValueError::type_factory()),
	  m_name_error(NameError::type_factory()), m_runtime_error(RuntimeError::type_factory()),
	  m_import_error(ImportError::type_factory()), m_key_error(KeyError::type_factory()),
	  m_not_implemented_error(NotImplementedError::type_factory()),
	  m_module_not_found_error(ModuleNotFoundError::type_factory()),
	  m_os_error(OSError::type_factory()), m_lookup_error(LookupError::type_factory()),
	  m_index_error(IndexError::type_factory()), m_warning(Warning::type_factory()),
	  m_deprecation_warning(DeprecationWarning::type_factory()),
	  m_import_warning(ImportWarning::type_factory()),
	  m_pending_deprecation_warning(PendingDeprecationWarning::type_factory()),
	  m_resource_warning(ResourceWarning::type_factory()),
	  m_syntax_error(SyntaxError::type_factory()), m_memory_error(MemoryError::type_factory()),
	  m_stop_iteration(StopIteration::type_factory()),
	  m_unbound_local_error(UnboundLocalError::type_factory())
{}

#define INITIALIZE_TYPE(TYPENAME)                                                         \
	PyType *TYPENAME()                                                                    \
	{                                                                                     \
		static PyType *type = nullptr;                                                    \
		if (!type) {                                                                      \
			auto &prototype = BuiltinTypes::the().TYPENAME();                             \
			type = PyType::initialize(prototype);                                         \
			spdlog::trace("Initialized builtin type {} @{}", type->name(), (void *)type); \
		}                                                                                 \
		return type;                                                                      \
	}

INITIALIZE_TYPE(type)
INITIALIZE_TYPE(super)

INITIALIZE_TYPE(bool_)
INITIALIZE_TYPE(bytes)
INITIALIZE_TYPE(bytes_iterator)
INITIALIZE_TYPE(bytearray)
INITIALIZE_TYPE(bytearray_iterator)
INITIALIZE_TYPE(ellipsis)
INITIALIZE_TYPE(str)
INITIALIZE_TYPE(str_iterator)
INITIALIZE_TYPE(float_)
INITIALIZE_TYPE(integer)
INITIALIZE_TYPE(complex)
INITIALIZE_TYPE(none)
INITIALIZE_TYPE(module)
INITIALIZE_TYPE(object)
INITIALIZE_TYPE(memoryview)

INITIALIZE_TYPE(dict)
INITIALIZE_TYPE(dict_items)
INITIALIZE_TYPE(dict_items_iterator)
INITIALIZE_TYPE(dict_keys)
INITIALIZE_TYPE(dict_key_iterator)
INITIALIZE_TYPE(dict_values)
INITIALIZE_TYPE(dict_value_iterator)
INITIALIZE_TYPE(mappingproxy)

INITIALIZE_TYPE(list)
INITIALIZE_TYPE(list_iterator)
INITIALIZE_TYPE(list_reverseiterator)

INITIALIZE_TYPE(tuple)
INITIALIZE_TYPE(tuple_iterator)

INITIALIZE_TYPE(set)
INITIALIZE_TYPE(frozenset)
INITIALIZE_TYPE(set_iterator)

INITIALIZE_TYPE(range)
INITIALIZE_TYPE(range_iterator)

INITIALIZE_TYPE(reversed)

INITIALIZE_TYPE(zip)
INITIALIZE_TYPE(map)

INITIALIZE_TYPE(enumerate)

INITIALIZE_TYPE(slice)

INITIALIZE_TYPE(iterator)

INITIALIZE_TYPE(function)
INITIALIZE_TYPE(native_function)
INITIALIZE_TYPE(llvm_function)
INITIALIZE_TYPE(code)
INITIALIZE_TYPE(cell)

INITIALIZE_TYPE(builtin_method)
INITIALIZE_TYPE(slot_wrapper)
INITIALIZE_TYPE(bound_method)
INITIALIZE_TYPE(method_wrapper)
INITIALIZE_TYPE(classmethod_descriptor)
INITIALIZE_TYPE(getset_descriptor)
INITIALIZE_TYPE(static_method)
INITIALIZE_TYPE(property)
INITIALIZE_TYPE(classmethod)
INITIALIZE_TYPE(member_descriptor)

INITIALIZE_TYPE(traceback)

INITIALIZE_TYPE(not_implemented)
INITIALIZE_TYPE(frame)

INITIALIZE_TYPE(namespace_)

INITIALIZE_TYPE(generator)
INITIALIZE_TYPE(coroutine)
INITIALIZE_TYPE(async_generator)

INITIALIZE_TYPE(generic_alias)

INITIALIZE_TYPE(base_exception)
INITIALIZE_TYPE(exception)
INITIALIZE_TYPE(type_error)
INITIALIZE_TYPE(assertion_error)
INITIALIZE_TYPE(attribute_error)
INITIALIZE_TYPE(value_error)
INITIALIZE_TYPE(name_error)
INITIALIZE_TYPE(runtime_error)
INITIALIZE_TYPE(import_error)
INITIALIZE_TYPE(key_error)
INITIALIZE_TYPE(not_implemented_error)
INITIALIZE_TYPE(module_not_found_error)
INITIALIZE_TYPE(os_error)
INITIALIZE_TYPE(lookup_error)
INITIALIZE_TYPE(index_error)
INITIALIZE_TYPE(warning)
INITIALIZE_TYPE(deprecation_warning)
INITIALIZE_TYPE(pending_deprecation_warning)
INITIALIZE_TYPE(resource_warning)
INITIALIZE_TYPE(import_warning)
INITIALIZE_TYPE(syntax_error)
INITIALIZE_TYPE(memory_error)
INITIALIZE_TYPE(stop_iteration)
INITIALIZE_TYPE(unbound_local_error)

}// namespace py::types
