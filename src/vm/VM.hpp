#pragma once

#include "executable/FunctionBlock.hpp"
#include "forward.hpp"
#include "memory/Heap.hpp"
#include "runtime/Value.hpp"
#include "utilities.hpp"

#include <stack>

class VirtualMachine;

using Registers = std::vector<py::Value>;

struct State
{
	std::optional<size_t> jump_block_count;
	bool catch_exception{ false };
};

struct StackFrame : NonCopyable
{
	Registers registers;
	Registers locals;
	InstructionBlock::const_iterator return_address;
	VirtualMachine *vm{ nullptr };
	std::unique_ptr<State> state;

	StackFrame() = delete;
	StackFrame(size_t register_count,
		size_t stack_size,
		InstructionBlock::const_iterator return_address,
		VirtualMachine *);
	StackFrame(StackFrame &&);
	~StackFrame();
};

class VirtualMachine
	: NonCopyable
	, NonMoveable
{
	std::stack<StackFrame> m_stack;
	std::deque<std::vector<const py::Value *>> m_stack_objects;

	InstructionBlock::const_iterator m_instruction_pointer;
	Heap &m_heap;
	std::unique_ptr<InterpreterSession> m_interpreter_session;
	State *m_state{ nullptr };

	friend StackFrame;

  public:
	int execute(std::shared_ptr<Program> program);

	static VirtualMachine &the()
	{
		static auto *vm = new VirtualMachine();
		return *vm;
	}

	py::Value &reg(size_t idx)
	{
		auto r = registers();
		ASSERT(r.has_value())
		ASSERT(idx < r->get().size())
		return r->get()[idx];
	}

	const py::Value &reg(size_t idx) const
	{
		auto r = registers();
		ASSERT(r.has_value())
		ASSERT(idx < r->get().size())
		return r->get()[idx];
	}

	py::Value &stack_local(size_t idx)
	{
		auto local = stack_locals();
		ASSERT(local.has_value())
		ASSERT(idx < local->get().size())
		return local->get()[idx];
	}

	const py::Value &stack_local(size_t idx) const
	{
		auto local = stack_locals();
		ASSERT(local.has_value())
		ASSERT(idx < local->get().size())
		return local->get()[idx];
	}

	std::optional<std::reference_wrapper<Registers>> registers()
	{
		if (!m_stack.empty()) { return m_stack.top().registers; }
		return {};
	}
	std::optional<std::reference_wrapper<const Registers>> registers() const
	{
		if (!m_stack.empty()) { return m_stack.top().registers; }
		return {};
	}

	std::optional<std::reference_wrapper<Registers>> stack_locals()
	{
		if (!m_stack.empty()) { return m_stack.top().locals; }
		return {};
	}

	std::optional<std::reference_wrapper<const Registers>> stack_locals() const
	{
		if (!m_stack.empty()) { return m_stack.top().locals; }
		return {};
	}

	const std::stack<StackFrame> &stack() const { return m_stack; }
	const State &state() const { return *m_state; }
	State &state() { return *m_state; }

	Heap &heap() { return m_heap; }

	Interpreter &interpreter();
	const Interpreter &interpreter() const;

	void set_instruction_pointer(InstructionBlock::const_iterator pos)
	{
		m_instruction_pointer = pos;
	}

	const InstructionBlock::const_iterator &instruction_pointer() const
	{
		return m_instruction_pointer;
	}

	void clear();

	void dump() const;

	void setup_call_stack(size_t register_count, size_t stack_size);
	int call(const std::shared_ptr<Function> &);
	void ret();
	void jump_blocks(size_t block_count);
	void set_exception_handling();

	void shutdown_interpreter(Interpreter &);

	const std::unique_ptr<InterpreterSession> &interpreter_session() const
	{
		return m_interpreter_session;
	}

	void push_frame(size_t register_count, size_t stack_size);

	void pop_frame();

	const std::deque<std::vector<const py::Value *>> &stack_objects() const
	{
		return m_stack_objects;
	}

  private:
	VirtualMachine();

	int execute_internal(std::shared_ptr<Program> program);

	void show_current_instruction(size_t index, size_t window) const;
};
