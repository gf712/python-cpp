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
		py::PyObject *exception{ nullptr };
	};

  protected:
	size_t m_register_count;
	py::PyModule *m_builtins;
	py::PyDict *m_globals;
	py::PyDict *m_locals;
	const py::PyTuple *m_consts;
	std::vector<py::PyCell *> m_freevars;
	ExecutionFrame *m_parent{ nullptr };
	py::PyObject *m_exception_to_catch{ nullptr };
	std::optional<ExceptionInfo> m_exception;
	std::optional<ExceptionInfo> m_stashed_exception;

  public:
	static ExecutionFrame *create(ExecutionFrame *parent,
		size_t register_count,
		size_t freevar_count,
		py::PyDict *globals,
		py::PyDict *locals,
		const py::PyTuple *consts);

	void put_local(const std::string &name, const py::Value &);
	void put_global(const std::string &name, const py::Value &);

	ExecutionFrame *parent() const { return m_parent; }

	void set_exception(py::PyObject *exception);

	void clear_stashed_exception();

	void stash_exception();

	const std::optional<ExceptionInfo> &exception_info() const { return m_exception; }

	const std::optional<ExceptionInfo> &stashed_exception_info() const
	{
		return m_stashed_exception;
	}

	bool catch_exception(py::PyObject *) const;

	void set_exception_to_catch(py::PyObject *exception);

	ExecutionFrame *exit();

	py::PyDict *globals() const;
	py::PyDict *locals() const;
	py::PyModule *builtins() const;
	const std::vector<py::PyCell *> &freevars() const;
	std::vector<py::PyCell *> &freevars();
	py::Value consts(size_t index) const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

  private:
	ExecutionFrame();
};