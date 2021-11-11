#pragma once

#include "PyObject.hpp"
#include "PyTuple.hpp"


class PyCode : public PyObject
{
	const std::shared_ptr<Function> m_function;
	const size_t m_function_id;
	const size_t m_register_count;
	const std::vector<std::string> m_args;
	PyModule *m_module;

  public:
	PyCode(std::shared_ptr<Function> function,
		size_t function_id,
		std::vector<std::string> args,
		PyModule *m_module);

	PyObject *call(PyTuple *args, PyDict *kwargs);
	const std::vector<std::string> &args() const { return m_args; }

	std::string to_string() const override { return fmt::format("PyCode"); }

	size_t register_count() const;

	const std::shared_ptr<Function> &function() const { return m_function; }

	void visit_graph(Visitor &) override;
};


class PyFunction : public PyObject
{
	const std::string m_name;
	PyCode *m_code;
	PyDict *m_globals;

  public:
	PyFunction(std::string, PyCode *code, PyDict *globals);

	const PyCode *code() const { return m_code; }

	std::string to_string() const override { return fmt::format("PyFunction"); }
	const std::string &name() const { return m_name; }

	PyDict *globals() const { return m_globals; }

	void visit_graph(Visitor &) override;
};


class PyNativeFunction : public PyObject
{
	std::string m_name;
	std::function<PyObject *(PyTuple *, PyDict *)> m_function;
	std::vector<PyObject *> m_captures;

  public:
	PyNativeFunction(std::string name, std::function<PyObject *(PyTuple *, PyDict *)> function)
		: PyObject(PyObjectType::PY_NATIVE_FUNCTION), m_name(std::move(name)),
		  m_function(std::move(function))
	{}

	// TODO: fix tracking of lambda captures
	template<typename... Args>
	PyNativeFunction(std::string name,
		std::function<PyObject *(PyTuple *, PyDict *)> function,
		Args &&... args)
		: PyObject(PyObjectType::PY_NATIVE_FUNCTION), m_name(std::move(name)),
		  m_function(std::move(function)), m_captures{ std::forward<Args>(args)... }
	{}

	PyObject *operator()(PyTuple *args, PyDict *kwargs) { return m_function(args, kwargs); }

	std::string to_string() const override
	{
		return fmt::format("PyNativeFunction {}", static_cast<const void *>(&m_function));
	}

	const std::string &name() const { return m_name; }

	void visit_graph(Visitor &) override;
};

template<> inline PyFunction *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_FUNCTION) { return static_cast<PyFunction *>(node); }
	return nullptr;
}

template<> inline const PyFunction *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_FUNCTION) { return static_cast<const PyFunction *>(node); }
	return nullptr;
}


template<> inline PyNativeFunction *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_NATIVE_FUNCTION) {
		return static_cast<PyNativeFunction *>(node);
	}
	return nullptr;
}


template<> inline const PyNativeFunction *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_NATIVE_FUNCTION) {
		return static_cast<const PyNativeFunction *>(node);
	}
	return nullptr;
}