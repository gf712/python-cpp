#pragma once

#include "forward.hpp"
#include "runtime/Value.hpp"

#include <unordered_map>
#include <memory>

struct SymbolTable
{
	std::unordered_map<std::string, std::shared_ptr<PyObject>> symbols;

	std::shared_ptr<PyObject> find(const std::string &name)
	{
		if (auto it = symbols.find(name); it != symbols.end()) {
			return it->second;
		} else {
			return nullptr;
		}
	}

	std::string to_string() const;
};

class ExecutionFrame
{
	std::unique_ptr<SymbolTable> m_symbol_table;
	std::array<Value, 16> m_parameters;
	std::shared_ptr<ExecutionFrame> m_parent{ nullptr };

  public:
	static std::shared_ptr<ExecutionFrame> create(std::shared_ptr<ExecutionFrame> parent)
	{
		auto new_frame = std::shared_ptr<ExecutionFrame>(new ExecutionFrame{});
		new_frame->m_parent = std::move(parent);
		return new_frame;
	}

	const Value &parameter(size_t parameter_idx) const
	{
		ASSERT(parameter_idx < m_parameters.size());
		return m_parameters[parameter_idx];
	}

	Value &parameter(size_t parameter_idx)
	{
		ASSERT(parameter_idx < m_parameters.size());
		return m_parameters[parameter_idx];
	}

	std::shared_ptr<PyObject> fetch_object(const std::string &name) const
	{
		if (auto obj = m_symbol_table->find(name)) { return obj; }
		if (!m_parent) {
			return nullptr;
		} else {
			return m_parent->fetch_object(name);
		}
	}

	void put_object(const std::string &name, std::shared_ptr<PyObject> obj)
	{
		m_symbol_table->symbols[name] = std::move(obj);
	}

	std::shared_ptr<ExecutionFrame> parent() const { return m_parent; }

	const std::unique_ptr<SymbolTable> &symbol_table() const { return m_symbol_table; }

  private:
	ExecutionFrame() : m_symbol_table(std::make_unique<SymbolTable>()) {}
};