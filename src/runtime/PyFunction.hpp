#pragma once

#include "PyObject.hpp"
#include "PyTuple.hpp"
#include "executable/Program.hpp"
#include "vm/VM.hpp"

namespace py {

class PyCode : public PyBaseObject
{
  public:
	const std::shared_ptr<Function> m_function;
	const size_t m_register_count;

	const std::vector<std::string> m_cellvars;
	const std::vector<std::string> m_varnames;
	const std::vector<std::string> m_freevars;
	const size_t m_stack_size;
	const std::string m_filename;
	const size_t m_first_line_number;
	const size_t m_arg_count;
	const size_t m_kwonly_arg_count;
	const std::vector<size_t> m_cell2arg;
	const size_t m_nlocals;
	CodeFlags m_flags;

  public:
	PyCode(std::shared_ptr<Function> function,
		std::vector<std::string> cellvars,
		std::vector<std::string> varnames,
		std::vector<std::string> freevars,
		size_t stack_size,
		std::string filename,
		size_t first_line_number,
		size_t arg_count,
		size_t kwonly_arg_count,
		std::vector<size_t> cell2arg,
		size_t nlocals,
		CodeFlags flags);

	PyObject *call(PyTuple *args, PyDict *kwargs);
	const std::vector<std::string> &varnames() const { return m_varnames; }

	std::string to_string() const override { return fmt::format("PyCode"); }

	size_t register_count() const;
	size_t freevars_count() const;
	size_t cellvars_count() const;
	size_t arg_count() const;
	size_t kwonly_arg_count() const;
	CodeFlags flags() const;
	const std::vector<size_t> &cell2arg() const;

	const std::shared_ptr<Function> &function() const { return m_function; }

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};


class PyFunction : public PyBaseObject
{
	const std::string m_name;
	PyCode *m_code;
	PyDict *m_globals;
	const std::vector<Value> m_defaults;
	const std::vector<Value> m_kwonly_defaults;
	std::vector<PyCell *> m_closure;
	PyModule *m_module = nullptr;

  public:
	PyFunction(std::string,
		std::vector<Value> defaults,
		std::vector<Value> kwonly_defaults,
		PyCode *code,
		std::vector<PyCell *> closure,
		PyDict *globals);

	const PyCode *code() const { return m_code; }

	std::string to_string() const override { return fmt::format("PyFunction"); }
	const std::vector<Value> &defaults() const { return m_defaults; }
	const std::vector<Value> &kwonly_defaults() const { return m_kwonly_defaults; }

	const std::string &name() const { return m_name; }

	PyObject *call_with_frame(PyDict *locals, PyTuple *args, PyDict *kwargs) const;

	PyObject *__call__(PyTuple *args, PyDict *kwargs);
	const std::string &function_name() const { return m_name; }

	PyDict *globals() const { return m_globals; }

	PyObject *__repr__() const;
	PyObject *__get__(PyObject *instance, PyObject *owner) const;

	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};


class PyNativeFunction : public PyBaseObject
{
	friend class ::Heap;

	std::string m_name;
	std::function<PyObject *(PyTuple *, PyDict *)> m_function;
	std::vector<PyObject *> m_captures;

	PyNativeFunction(std::string &&name, std::function<PyObject *(PyTuple *, PyDict *)> &&function);

	// TODO: fix tracking of lambda captures
	template<typename... Args>
	PyNativeFunction(std::string &&name,
		std::function<PyObject *(PyTuple *, PyDict *)> &&function,
		Args &&... args)
		: PyNativeFunction(std::move(name), std::move(function))
	{
		m_captures = std::vector<PyObject *>{ std::forward<Args>(args)... };
	}

  public:
	template<typename... Args>
	static PyNativeFunction *create(std::string name,
		std::function<PyObject *(PyTuple *, PyDict *)> function,
		Args &&... args)
	{
		return VirtualMachine::the().heap().allocate<PyNativeFunction>(
			std::move(name), std::move(function), std::forward<Args>(args)...);
	}

	PyObject *operator()(PyTuple *args, PyDict *kwargs) { return m_function(args, kwargs); }

	std::string to_string() const override
	{
		return fmt::format("PyNativeFunction {}", static_cast<const void *>(&m_function));
	}

	const std::string &name() const { return m_name; }
	PyObject *__call__(PyTuple *args, PyDict *kwargs);
	PyObject *__repr__() const;

	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py