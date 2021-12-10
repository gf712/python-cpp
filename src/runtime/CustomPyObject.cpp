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

namespace {
template<typename T, typename... U> size_t get_address(std::function<T(U...)> f)
{
	// adapted from https://stackoverflow.com/a/35920804
	using FunctionType = T (*)(U...);
	auto fn_ptr = f.template target<FunctionType>();
	return bit_cast<size_t>(*fn_ptr);
}
}// namespace

CustomPyObject::CustomPyObject(const PyType *type)
	: PyBaseObject(type->underlying_type()), m_type_obj(type)
{}

PyObject *CustomPyObject::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	if ((args && !args->elements().empty()) || (kwargs && !kwargs->map().empty())) {

		if (!type->underlying_type().__dict__->map().contains(String{ "__new__" })) {
			ASSERT(type->underlying_type().__new__)
			const auto new_fn = get_address(*type->underlying_type().__new__);
			ASSERT(new_fn)

			ASSERT(custom_object()->underlying_type().__new__)
			const auto custom_new_fn = get_address(*custom_object()->underlying_type().__new__);
			ASSERT(custom_new_fn)

			if (new_fn != custom_new_fn) {
				VirtualMachine::the().interpreter().raise_exception(type_error(
					"object.__new__() takes exactly one argument (the type to instantiate)"));
				return nullptr;
			}
		}

		if (!type->underlying_type().__dict__->map().contains(String{ "__init__" })) {
			ASSERT(type->underlying_type().__init__)
			const auto init_fn = get_address(*type->underlying_type().__init__);
			ASSERT(init_fn)

			ASSERT(custom_object()->underlying_type().__init__)
			const auto custom_init_fn = get_address(*custom_object()->underlying_type().__init__);
			ASSERT(custom_init_fn)
			if (init_fn == custom_init_fn) {
				VirtualMachine::the().interpreter().raise_exception(
					type_error("object() takes no arguments"));
				return nullptr;
			}
		}
	}
	return VirtualMachine::the().heap().allocate<CustomPyObject>(type);
}

std::optional<int32_t> CustomPyObject::__init__(PyTuple *args, PyDict *kwargs)
{
	if ((args && !args->elements().empty()) || (kwargs && !kwargs->map().empty())) {
		if (!m_type_obj->underlying_type().__dict__->map().contains(String{ "__new__" })) {
			ASSERT(m_type_obj->underlying_type().__new__)
			const auto new_fn = get_address(*m_type_obj->underlying_type().__new__);
			ASSERT(new_fn)

			ASSERT(custom_object()->underlying_type().__new__)
			const auto custom_new_fn = get_address(*custom_object()->underlying_type().__new__);
			ASSERT(custom_new_fn)

			if (new_fn == custom_new_fn) {
				VirtualMachine::the().interpreter().raise_exception(type_error(
					"object.__new__() takes exactly one argument (the type to instantiate)"));
				return -1;
			}
		}

		if (!m_type_obj->underlying_type().__dict__->map().contains(String{ "__init__" })) {
			ASSERT(m_type_obj->underlying_type().__init__)
			const auto init_fn = get_address(*m_type_obj->underlying_type().__init__);
			ASSERT(init_fn)

			ASSERT(custom_object()->underlying_type().__init__)
			const auto custom_init_fn = get_address(*custom_object()->underlying_type().__init__);
			ASSERT(custom_init_fn)

			if (init_fn != custom_init_fn) {
				VirtualMachine::the().interpreter().raise_exception(
					type_error("object() takes no arguments"));
				return -1;
			}
		}
	}
	return 0;
}

PyObject *CustomPyObject::__repr__() const
{
	return PyString::create(
		fmt::format("{} object at {}", m_type_prototype.__name__, static_cast<const void *>(this)));
}


PyType *CustomPyObject::type() const
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