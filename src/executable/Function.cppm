module;

#include "core.hpp"
#include <cstddef>
#include <cstdint>

export module py.runtime:executable_function;
import :value;
import std;

export class Program;

export class VirtualMachine;

enum class FunctionExecutionBackend { BYTECODE = 0, LLVM = 1 };

export class Interpreter;

class Function : NonCopyable
{
  protected:
	std::size_t m_register_count;
	std::size_t m_locals_count;
	std::size_t m_stack_size;
	std::string m_function_name;
	FunctionExecutionBackend m_backend;
	std::shared_ptr<Program> m_program;

  public:
	Function(std::size_t register_count,
		std::size_t locals_count,
		std::size_t stack_size,
		std::string function_name,
		FunctionExecutionBackend backend,
		std::shared_ptr<Program> program)
		: m_register_count(register_count), m_locals_count(locals_count), m_stack_size(stack_size),
		  m_function_name(function_name), m_backend(backend), m_program(std::move(program))
	{}
	virtual ~Function() = default;

	std::size_t register_count() const { return m_register_count; }
	std::size_t locals_count() const { return m_locals_count; }
	std::size_t stack_size() const { return m_stack_size; }

	FunctionExecutionBackend backend() const { return m_backend; }

	const std::string function_name() const { return m_function_name; }

	virtual std::string to_string() const = 0;

	virtual std::vector<std::uint8_t> serialize() const = 0;

	std::shared_ptr<Program> program() const { return m_program; }

	virtual py::PyResult<py::Value> call(VirtualMachine &, Interpreter &) const = 0;
	virtual py::PyResult<py::Value> call_without_setup(VirtualMachine &, Interpreter &) const = 0;
};
