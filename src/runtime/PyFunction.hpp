#pragma once

#include "MemoryError.hpp"
#include "PyObject.hpp"
#include "PyTuple.hpp"
#include "executable/Program.hpp"
#include "vm/VM.hpp"

namespace py {
class PyFunction : public PyBaseObject
{
	friend class ::Heap;

	PyString *m_name = nullptr;
	PyString *m_doc = nullptr;
	PyCode *m_code = nullptr;
	PyDict *m_globals = nullptr;
	PyDict *m_dict = nullptr;
	const std::vector<Value> m_defaults;
	const std::vector<Value> m_kwonly_defaults;
	PyTuple *m_closure{ nullptr };
	PyString *m_module = nullptr;

	PyFunction(PyType *);

  public:
	PyFunction(std::string,
		std::vector<Value> defaults,
		std::vector<Value> kwonly_defaults,
		PyCode *code,
		PyTuple *closure,
		PyDict *globals);

	const PyCode *code() const { return m_code; }
	PyCode *code() { return m_code; }

	std::string to_string() const override;
	const std::vector<Value> &defaults() const { return m_defaults; }
	const std::vector<Value> &kwonly_defaults() const { return m_kwonly_defaults; }

	PyResult<PyObject *> call_with_frame(PyDict *locals, PyTuple *args, PyDict *kwargs) const;

	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);
	PyString *function_name() const { return m_name; }

	PyDict *globals() const { return m_globals; }

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __get__(PyObject *instance, PyObject *owner) const;

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};


class PyNativeFunction : public PyBaseObject
{
	friend class ::Heap;
	using FreeFunctionSignature = PyResult<PyObject *>(PyTuple *, PyDict *);
	using MethodSignature = PyResult<PyObject *>(PyObject *, PyTuple *, PyDict *);
	using FreeFunctionType = std::function<FreeFunctionSignature>;
	using MethodType = std::function<MethodSignature>;
	using FunctionType = std::variant<FreeFunctionType, MethodType>;

	using FreeFunctionPointerType = typename std::add_pointer_t<FreeFunctionSignature>;
	using MethodPointerType = typename std::add_pointer_t<MethodSignature>;

	std::string m_name;
	FunctionType m_function;
	PyObject *m_self{ nullptr };
	std::vector<PyObject *> m_captures;

	PyNativeFunction(PyType *);

	PyNativeFunction(std::string &&name, FunctionType &&function);

	// TODO: fix tracking of lambda captures
	template<typename... Args>
	PyNativeFunction(std::string &&name, FunctionType &&function, PyObject *self, Args &&...args)
		: PyNativeFunction(std::move(name), std::move(function))
	{
		m_self = self;
		m_captures = std::vector<PyObject *>{ std::forward<Args>(args)... };
	}

  public:
	template<typename... Args>
	static PyResult<PyNativeFunction *>
		create(std::string name, FreeFunctionType function, Args &&...args)
	{
		auto *result = VirtualMachine::the().heap().allocate<PyNativeFunction>(
			std::move(name), std::move(function), nullptr, std::forward<Args>(args)...);
		if (!result) { return Err(memory_error(sizeof(PyNativeFunction))); }
		return Ok(result);
	}

	template<typename... Args>
	static PyResult<PyNativeFunction *>
		create(std::string name, MethodType function, PyObject *self, Args &&...args)
	{
		auto *result = VirtualMachine::the().heap().allocate<PyNativeFunction>(
			std::move(name), std::move(function), self, std::forward<Args>(args)...);
		if (!result) { return Err(memory_error(sizeof(PyNativeFunction))); }
		return Ok(result);
	}

	PyResult<PyObject *> operator()(PyTuple *args, PyDict *kwargs)
	{
		ASSERT(is_function());
		return std::get<FreeFunctionType>(m_function)(args, kwargs);
	}

	PyResult<PyObject *> operator()(PyObject *self, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(is_method());
		return std::get<MethodType>(m_function)(self, args, kwargs);
	}

	bool is_function() const { return std::holds_alternative<FreeFunctionType>(m_function); }

	bool is_method() const { return std::holds_alternative<MethodType>(m_function); }

	std::optional<FreeFunctionPointerType> free_function_pointer()
	{
		if (!is_function()) { return std::nullopt; }
		ASSERT(std::get<FreeFunctionType>(m_function).target<FreeFunctionPointerType>());
		return *std::get<FreeFunctionType>(m_function).target<FreeFunctionPointerType>();
	}

	std::optional<MethodPointerType> method_pointer()
	{
		if (!is_method()) { return std::nullopt; }
		ASSERT(std::get<MethodType>(m_function).target<MethodPointerType>());
		return *std::get<MethodType>(m_function).target<MethodPointerType>();
	}

	std::string to_string() const override;

	const std::string &name() const { return m_name; }
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;

	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
