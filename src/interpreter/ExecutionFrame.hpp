#pragma once

#include "forward.hpp"
#include "vm/VM.hpp"

#include "runtime/Value.hpp"

#include <memory>
#include <unordered_map>


class ExecutionFrame : public Cell
{
	friend Heap;
	friend Interpreter;

	struct ExceptionInfo
	{
		PyObject *exception{ nullptr };
	};

  protected:
	// parameters
	std::array<std::optional<Value>, 16> m_parameters;
	size_t m_register_count;
	PyModule *m_builtins;
	PyDict *m_globals;
	PyDict *m_locals;
	ExecutionFrame *m_parent{ nullptr };
	PyObject *m_exception_to_catch{ nullptr };
	std::optional<ExceptionInfo> m_exception;
	std::optional<ExceptionInfo> m_stashed_exception;

  public:
	static ExecutionFrame *
		create(ExecutionFrame *parent, size_t register_count, PyDict *globals, PyDict *locals);

	const std::optional<Value> &parameter(size_t parameter_idx) const
	{
		ASSERT(parameter_idx < m_parameters.size());
		return m_parameters[parameter_idx];
	}

	std::optional<Value> &parameter(size_t parameter_idx)
	{
		ASSERT(parameter_idx < m_parameters.size());
		return m_parameters[parameter_idx];
	}

	void put_local(const std::string &name, PyObject *obj);
	void put_global(const std::string &name, PyObject *obj);

	ExecutionFrame *parent() const { return m_parent; }

	void set_exception(PyObject *exception);

	void clear_stashed_exception();

	void stash_exception();

	const std::optional<ExceptionInfo> &exception_info() const { return m_exception; }

	const std::optional<ExceptionInfo> &stashed_exception_info() const
	{
		return m_stashed_exception;
	}

	bool catch_exception(PyObject *) const;

	void set_exception_to_catch(PyObject *exception);

	ExecutionFrame *exit();

	PyDict *globals() const;
	PyDict *locals() const;
	PyModule *builtins() const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

  private:
	ExecutionFrame();
};