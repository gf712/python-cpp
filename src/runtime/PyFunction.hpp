#pragma once

#include "PyObject.hpp"
#include "PyTuple.hpp"
#include "executable/Program.hpp"
#include "vm/VM.hpp"

namespace py {
class PyFunction : public PyBaseObject
{
	friend std::unique_ptr<TypePrototype> register_type();

	PyString *m_name = nullptr;
	PyCode *m_code = nullptr;
	PyDict *m_globals = nullptr;
	PyDict *m_dict = nullptr;
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

	PyObject *call_with_frame(PyDict *locals, PyTuple *args, PyDict *kwargs) const;

	PyObject *__call__(PyTuple *args, PyDict *kwargs);
	PyString *function_name() const { return m_name; }

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

	std::string to_string() const override;

	const std::string &name() const { return m_name; }
	PyObject *__call__(PyTuple *args, PyDict *kwargs);
	PyObject *__repr__() const;

	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py