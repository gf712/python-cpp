#pragma once

#include "MemoryError.hpp"
#include "PyObject.hpp"
#include "vm/VM.hpp"

namespace py {

// Essentially the same as PyNativeFunction, but wraps around a std::function
// with return type PyObject*, instead of PyResult. This makes it easier to interop
// between Python and LLVM, since PyResult is not an interop type (it's a variant,
// and it doesn't make sense to force all functions to return PyResult by pointer).
class PyLLVMFunction : public PyBaseObject
{
	friend class ::Heap;
	using FunctionType = std::function<PyObject *(PyTuple *, PyDict *)>;

	std::string m_name;
	FunctionType m_function;
	std::vector<PyObject *> m_captures;

	PyLLVMFunction(std::string &&name, FunctionType &&function);

	// TODO: fix tracking of lambda captures
	template<typename... Args>
	PyLLVMFunction(std::string &&name, FunctionType &&function, Args &&... args)
		: PyLLVMFunction(std::move(name), std::move(function))
	{
		m_captures = std::vector<PyObject *>{ std::forward<Args>(args)... };
	}

  public:
	template<typename... Args>
	static PyResult<PyLLVMFunction *>
		create(std::string name, FunctionType &&function, Args &&... args)
	{
		auto *result = VirtualMachine::the().heap().allocate<PyLLVMFunction>(
			std::move(name), std::move(function), std::forward<Args>(args)...);
		if (!result) { return Err(memory_error(sizeof(PyLLVMFunction))); }
		return Ok(result);
	}

	PyResult<PyObject *> operator()(PyTuple *args, PyDict *kwargs)
	{
		// TODO: deal with exceptions in llvm land
		return Ok(m_function(args, kwargs));
	}

	std::string to_string() const override;

	const std::string &name() const { return m_name; }
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;

	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};
}// namespace py