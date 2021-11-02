#pragma once

#include "forward.hpp"
#include "bytecode/VM.hpp"

#include "runtime/Value.hpp"

#include <unordered_map>
#include <memory>


class ExecutionFrame : public Cell
{
	friend Heap;

	// parameters
	std::array<std::optional<Value>, 16> m_parameters;
	PyModule *m_builtins;
	PyDict *m_globals;
	PyDict *m_locals;
	PyDict *m_ns;
	ExecutionFrame *m_parent{ nullptr };
	size_t m_return_address;
	std::optional<LocalFrame> m_frame_info;
	PyObject *m_exception{ nullptr };
	PyObject *m_exception_to_catch{ nullptr };

  public:
	static ExecutionFrame *
		create(ExecutionFrame *parent, PyDict *globals, PyDict *locals, PyDict *ns);

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

	void set_return_address(size_t address) { m_return_address = address; }
	size_t return_address() const { return m_return_address; }

	void attach_frame(LocalFrame &&frame) { m_frame_info.emplace(std::move(frame)); }

	void set_exception(PyObject *exception);

	PyObject *exception() const { return m_exception; }

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