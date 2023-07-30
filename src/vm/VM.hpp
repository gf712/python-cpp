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
	enum class CleanupLogic {
		CATCH_EXCEPTION,
		WITH_EXIT,
	};
	std::stack<std::optional<std::pair<CleanupLogic, InstructionVector::const_iterator>>> cleanup{
		{ std::nullopt }
	};
};

struct StackFrame : NonCopyable
{
  private:
	StackFrame() = default;

	StackFrame(size_t register_count,
		size_t stack_size,
		InstructionVector::const_iterator return_address,
		VirtualMachine *);

	StackFrame(StackFrame &&);

  public:
	template<typename... Args> static std::unique_ptr<StackFrame> create(Args &&...args)
	{
		return std::unique_ptr<StackFrame>(new StackFrame{ std::forward<Args>(args)... });
	}

	Registers registers;
	Registers locals;
	InstructionVector::const_iterator return_address;
	InstructionVector::const_iterator last_instruction_pointer;
	VirtualMachine *vm{ nullptr };
	std::unique_ptr<State> state;

	~StackFrame();

	StackFrame clone() const;

	StackFrame &restore();
	void leave();
};

class VirtualMachine
	: NonCopyable
	, NonMoveable
{
	std::stack<std::reference_wrapper<StackFrame>> m_stack;
	std::deque<std::vector<const py::Value *>> m_stack_objects;

	InstructionVector::const_iterator m_instruction_pointer;
	std::unique_ptr<Interpreter> m_interpreter;
	std::unique_ptr<Heap> m_heap;
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
		if (!m_stack.empty()) { return m_stack.top().get().registers; }
		return {};
	}
	std::optional<std::reference_wrapper<const Registers>> registers() const
	{
		if (!m_stack.empty()) { return m_stack.top().get().registers; }
		return {};
	}

	std::optional<std::reference_wrapper<Registers>> stack_locals()
	{
		if (!m_stack.empty()) { return m_stack.top().get().locals; }
		return {};
	}

	std::optional<std::reference_wrapper<const Registers>> stack_locals() const
	{
		if (!m_stack.empty()) { return m_stack.top().get().locals; }
		return {};
	}

	const std::stack<std::reference_wrapper<StackFrame>> &stack() const { return m_stack; }
	const State &state() const { return *m_state; }
	State &state() { return *m_state; }

	Heap &heap() { return *m_heap; }

	Interpreter &initialize_interpreter(std::shared_ptr<Program> &&);
	Interpreter &interpreter();
	const Interpreter &interpreter() const;
	bool has_interpreter() const { return m_interpreter.operator bool(); }

	void set_instruction_pointer(InstructionVector::const_iterator pos)
	{
		m_instruction_pointer = pos;
		m_stack.top().get().last_instruction_pointer = m_instruction_pointer;
	}

	const InstructionVector::const_iterator &instruction_pointer() const
	{
		return m_instruction_pointer;
	}

	const py::Value *stack_pointer() const
	{
		ASSERT(!m_stack.empty())
		ASSERT(!m_stack.top().get().registers.empty())
		return &m_stack.top().get().registers.front();
	}

	void clear();

	void dump() const;

	[[nodiscard]] std::unique_ptr<StackFrame> setup_call_stack(size_t register_count,
		size_t stack_size);
	int call(const std::unique_ptr<Function> &);
	void ret();
	void set_cleanup(State::CleanupLogic cleanup_type,
		InstructionVector::const_iterator exit_instruction);
	void leave_cleanup_handling();

	std::unique_ptr<StackFrame> push_frame(size_t register_count, size_t stack_size);
	void push_frame(StackFrame &frame);

	void pop_frame();

	const std::deque<std::vector<const py::Value *>> &stack_objects() const
	{
		return m_stack_objects;
	}

	py::PyModule *import(py::PyString *path);

  private:
	VirtualMachine();

	int execute_internal(std::shared_ptr<Program> program);

	void show_current_instruction(size_t index, size_t window) const;
};
