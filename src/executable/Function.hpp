#pragma once

#include "FunctionBlock.hpp"

enum class FunctionExecutionBackend { BYTECODE = 0, LLVM = 1 };

class Function : NonCopyable
{
  protected:
	size_t m_register_count;
	size_t m_locals_count;
	size_t m_stack_size;
	std::string m_function_name;
	FunctionExecutionBackend m_backend;
	std::shared_ptr<Program> m_program;

  public:
	Function(size_t register_count,
		size_t locals_count,
		size_t stack_size,
		std::string function_name,
		FunctionExecutionBackend backend,
		std::shared_ptr<Program> program)
		: m_register_count(register_count), m_locals_count(locals_count), m_stack_size(stack_size),
		  m_function_name(function_name), m_backend(backend), m_program(std::move(program))
	{}
	virtual ~Function() = default;

	size_t register_count() const { return m_register_count; }
	size_t locals_count() const { return m_locals_count; }
	size_t stack_size() const { return m_stack_size; }

	FunctionExecutionBackend backend() const { return m_backend; }

	const std::string function_name() const { return m_function_name; }

	virtual std::string to_string() const = 0;

	virtual std::vector<uint8_t> serialize() const = 0;

	std::shared_ptr<Program> program() const { return m_program; }

	virtual py::PyResult<py::Value> call(VirtualMachine &, Interpreter &) const = 0;
	virtual py::PyResult<py::Value> call_without_setup(VirtualMachine &, Interpreter &) const = 0;
};