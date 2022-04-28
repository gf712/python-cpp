#pragma once

#include "Exception.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

namespace py {

class MemoryError : public Exception
{
	friend class ::Heap;
	friend BaseException *memory_error(size_t failed_allocation_size);

  private:
	MemoryError(PyTuple *args);

	static PyResult create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		auto result = heap.allocate<MemoryError>(args);
		if (!result) {
			// TODO: if this exception fails to allocated we need to find a solution to signal it.
			//       could force a GC run and try again?
			TODO();
		}
		return PyResult::Ok(result);
	}

  public:
	static PyType *register_type(PyModule *);

	static PyResult __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyType *type() const override;

	static PyType *this_type();

	std::string to_string() const override;
};

template<> MemoryError *as(PyObject *obj);
template<> const MemoryError *as(const PyObject *obj);

inline BaseException *memory_error(size_t failed_allocation_size)
{
	// if the allocation of the exception parameters fail, we bail (for now at least)
	auto msg = PyString::create(
		fmt::format("memory allocation failed, allocating {} bytes", failed_allocation_size));
	if (msg.is_err()) { TODO(); }
	auto args_tuple = PyTuple::create();
	if (args_tuple.is_err()) { TODO(); }
	return MemoryError::create(args_tuple.unwrap_as<PyTuple>()).unwrap_as<MemoryError>();
}

}// namespace py