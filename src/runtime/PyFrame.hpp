#pragma once

#include "forward.hpp"
#include "vm/VM.hpp"

#include "runtime/Value.hpp"

#include <memory>
#include <unordered_map>

namespace py {

class PyFrame : public PyBaseObject
{
	friend Heap;
	friend Interpreter;

	struct ExceptionStackItem
	{
		py::BaseException *exception{ nullptr };
		py::PyType *exception_type{ nullptr };
		py::PyTraceback *traceback{ nullptr };
	};

  protected:
	// next outer frame object (this frame’s caller)
	PyFrame *m_f_back{ nullptr };
	// builtins namespace seen by this frame
	py::PyModule *m_builtins;
	// global namespace seen by this frame
	py::PyDict *m_globals;
	// local namespace seen by this frame
	py::PyDict *m_locals;
	size_t m_register_count;
	const py::PyTuple *m_consts;
	std::vector<py::PyCell *> m_freevars;
	py::BaseException *m_exception_to_catch{ nullptr };
	std::vector<ExceptionStackItem> m_exception_stack;

  public:
	static PyFrame *create(PyFrame *parent,
		size_t register_count,
		size_t freevar_count,
		py::PyDict *globals,
		py::PyDict *locals,
		const py::PyTuple *consts);

	void put_local(const std::string &name, const py::Value &);
	void put_global(const std::string &name, const py::Value &);

	PyFrame *parent() const { return m_f_back; }

	void push_exception(py::BaseException *exception);
	py::BaseException *pop_exception();

	std::optional<ExceptionStackItem> exception_info() const
	{
		if (m_exception_stack.empty()) return {};
		return m_exception_stack.back();
	}

	bool catch_exception(py::PyObject *) const;

	void set_exception_to_catch(py::BaseException *exception);

	PyFrame *exit();

	py::PyDict *globals() const;
	py::PyDict *locals() const;
	py::PyModule *builtins() const;
	const std::vector<py::PyCell *> &freevars() const;
	std::vector<py::PyCell *> &freevars();
	py::Value consts(size_t index) const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();

  private:
	PyFrame();
};

}// namespace py