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
	template<typename T> friend class GeneratorInterface;

  public:
	struct ExceptionStackItem
	{
		BaseException *exception{ nullptr };
		PyType *exception_type{ nullptr };
		PyTraceback *traceback{ nullptr };
	};

  private:
	PyFrame(const std::vector<std::string> &);

  protected:
	// next outer frame object (this frame’s caller)
	PyFrame *m_f_back{ nullptr };
	// builtins namespace seen by this frame
	PyModule *m_builtins{ nullptr };
	// global namespace seen by this frame
	PyDict *m_globals{ nullptr };
	// local namespace seen by this frame
	PyDict *m_locals{ nullptr };
	// code segment
	PyCode *m_f_code{ nullptr };
	// generator object
	PyObject *m_generator{ nullptr };

	size_t m_register_count;
	const std::vector<std::string> &m_names;
	const PyTuple *m_consts;
	std::vector<PyCell *> m_freevars;
	BaseException *m_exception_to_catch{ nullptr };
	std::shared_ptr<std::vector<ExceptionStackItem>> m_exception_stack;

  public:
	static PyFrame *create(PyFrame *parent,
		size_t register_count,
		size_t freevar_count,
		PyCode *code,
		PyDict *globals,
		PyDict *locals,
		const PyTuple *consts,
		const std::vector<std::string> &names,
		PyObject *generator);

	void put_local(const std::string &name, const Value &);
	void put_global(const std::string &name, const Value &);

	PyFrame *parent() const { return m_f_back; }

	void push_exception(BaseException *exception);
	BaseException *pop_exception();

	std::optional<ExceptionStackItem> exception_info() const
	{
		if (m_exception_stack->empty()) return {};
		return m_exception_stack->back();
	}

	bool catch_exception(PyObject *) const;

	void set_exception_to_catch(BaseException *exception);

	PyFrame *exit();

	PyDict *globals() const;
	PyDict *locals() const;
	PyModule *builtins() const;
	PyCode *code() const { return m_f_code; }
	PyObject *generator() const { return m_generator; }
	void set_generator(PyObject *generator) { m_generator = generator; }

	const std::vector<PyCell *> &freevars() const;
	std::vector<PyCell *> &freevars();
	Value consts(size_t index) const;
	const std::string &names(size_t index) const;

	std::string to_string() const override;
	void visit_graph(Visitor &) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *type() const override;

  private:
	PyFrame();
};

}// namespace py