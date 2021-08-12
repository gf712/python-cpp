#pragma once

#include "forward.hpp"
#include "bytecode/VM.hpp"

#include "runtime/Value.hpp"

#include <unordered_map>
#include <memory>


class ExecutionFrame
{
	// parameters
	std::array<std::optional<Value>, 16> m_parameters;
	std::shared_ptr<PyModule> m_builtins;
	std::shared_ptr<PyDict> m_globals;
	std::unique_ptr<PyDict> m_locals;
	std::shared_ptr<PyDict> m_ns;
	std::shared_ptr<ExecutionFrame> m_parent{ nullptr };
	size_t m_return_address;
	std::optional<LocalFrame> m_frame_info;
	std::shared_ptr<PyObject> m_exception{ nullptr };
	std::shared_ptr<PyObject> m_exception_to_catch{ nullptr };

  public:
	static std::shared_ptr<ExecutionFrame> create(std::shared_ptr<ExecutionFrame> parent,
		std::shared_ptr<PyDict> globals,
		std::unique_ptr<PyDict> &&locals,
		std::shared_ptr<PyDict> ns);

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

	void put_local(const std::string &name, std::shared_ptr<PyObject> obj);
	void put_global(const std::string &name, std::shared_ptr<PyObject> obj);

	std::shared_ptr<ExecutionFrame> parent() const { return m_parent; }

	void set_return_address(size_t address) { m_return_address = address; }
	size_t return_address() const { return m_return_address; }

	void attach_frame(LocalFrame &&frame) { m_frame_info.emplace(std::move(frame)); }

	void set_exception(std::shared_ptr<PyObject> exception);

	std::shared_ptr<PyObject> exception() const { return m_exception; }

	bool catch_exception(std::shared_ptr<PyObject>) const;

	void set_exception_to_catch(std::shared_ptr<PyObject> exception);

	std::shared_ptr<ExecutionFrame> exit();

	const std::shared_ptr<PyDict> &globals() const;
	const std::unique_ptr<PyDict> &locals() const;
	const std::shared_ptr<PyModule> &builtins() const;

  private:
	ExecutionFrame();
};