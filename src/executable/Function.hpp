#pragma once

#include "FunctionBlock.hpp"

class View
{
	InstructionVector::const_iterator m_begin;
	InstructionVector::const_iterator m_end;

  public:
	View(InstructionVector::const_iterator begin, InstructionVector::const_iterator end)
		: m_begin(begin), m_end(end)
	{}

	auto begin() const { return m_begin; }
	auto end() const { return m_end; }
};

enum class FunctionExecutionBackend { BYTECODE = 0, LLVM = 1 };

class Function
{
  protected:
	size_t m_register_count;
	size_t m_stack_size;
	std::string m_function_name;
	FunctionExecutionBackend m_backend;

  public:
	Function(size_t register_count,
		size_t stack_size,
		std::string function_name,
		FunctionExecutionBackend backend)
		: m_register_count(register_count), m_stack_size(stack_size),
		  m_function_name(function_name), m_backend(backend)
	{}
	virtual ~Function() = default;

	size_t register_count() const { return m_register_count; }
	size_t stack_size() const { return m_stack_size; }

	FunctionExecutionBackend backend() const { return m_backend; }

	const std::string function_name() const { return m_function_name; }

	virtual std::string to_string() const = 0;

	virtual std::vector<uint8_t> serialize() const = 0;

	virtual py::PyResult<py::Value> call(VirtualMachine &, Interpreter &) const = 0;
};