#include "KeyError.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> KeyError *as(PyObject *obj)
{
	ASSERT(types::key_error())
	if (obj->type() == types::key_error()) { return static_cast<KeyError *>(obj); }
	return nullptr;
}

template<> const KeyError *as(const PyObject *obj)
{
	ASSERT(types::key_error())
	if (obj->type() == types::key_error()) { return static_cast<const KeyError *>(obj); }
	return nullptr;
}

KeyError::KeyError(PyType *type) : Exception(type) {}

KeyError::KeyError(PyTuple *args) : Exception(types::BuiltinTypes::the().key_error(), args) {}

PyResult<KeyError *> KeyError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto result = heap.allocate<KeyError>(args);
	if (!result) { return Err(memory_error(sizeof(KeyError))); }
	return Ok(result);
}

PyResult<PyObject *> KeyError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::key_error())
	ASSERT(!kwargs || kwargs->map().empty())
	return KeyError::create(args);
}

PyType *KeyError::static_type() const
{
	ASSERT(types::key_error())
	return types::key_error();
}

PyType *KeyError::class_type()
{
	ASSERT(types::key_error())
	return types::key_error();
}

namespace {

	std::once_flag key_error_flag;

	std::unique_ptr<TypePrototype> register_key_error()
	{
		return std::move(klass<KeyError>("KeyError", Exception::class_type()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> KeyError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(key_error_flag, []() { type = register_key_error(); });
		return std::move(type);
	};
}

}// namespace py
