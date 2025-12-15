#pragma once

#include "Exception.hpp"
#include "ImportError.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class ModuleNotFoundError : public ImportError
{
	friend class ::Heap;
	template<typename... Args>
	friend BaseException *module_not_found_error(const std::string &message, Args &&...args);

	PyObject *m_name{ nullptr };
	PyObject *m_path{ nullptr };

  private:
	ModuleNotFoundError(PyTuple *msg, PyObject *name, PyObject *path);

	ModuleNotFoundError(PyType *type);

	static ModuleNotFoundError *create(PyTuple *args, PyObject *name, PyObject *path)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<ModuleNotFoundError>(args, name, path);
	}

  public:
	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyType *static_type() const override;
};

template<typename... Args>
inline BaseException *module_not_found_error(const std::string &message, Args &&...args)
{
	auto msg = PyString::create(fmt::format(message, std::forward<Args>(args)...));
	ASSERT(msg.is_ok());
	auto args_tuple = PyTuple::create(msg.unwrap());
	ASSERT(args_tuple.is_ok());
	return ModuleNotFoundError::create(args_tuple.unwrap(), py_none(), py_none());
}

}// namespace py
