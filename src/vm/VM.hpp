#pragma once

#include "executable/FunctionBlock.hpp"
#include "forward.hpp"
#include "memory/Heap.hpp"
#include "runtime/Value.hpp"
#include "utilities.hpp"

#include <stack>

class VirtualMachine;

using Registers = std::vector<Value>;

struct StackFrame : NonCopyable
{
	Registers registers;
	InstructionVector::const_iterator return_address;
	VirtualMachine *vm{ nullptr };

	StackFrame(size_t frame_size,
		InstructionVector::const_iterator return_address,
		VirtualMachine *);
	StackFrame(StackFrame &&);
	~StackFrame();
};

class VirtualMachine
	: NonCopyable
	, NonMoveable
{
	std::stack<StackFrame> m_stack;
	InstructionVector::const_iterator m_instruction_pointer;
	Heap &m_heap;
	std::unique_ptr<InterpreterSession> m_interpreter_session;

	friend StackFrame;

  public:
	int execute(std::shared_ptr<Program> program);

	static VirtualMachine &the()
	{
		static auto *vm = new VirtualMachine();
		return *vm;
	}

	Value &reg(size_t idx)
	{
		ASSERT(idx < registers().size())
		return registers()[idx];
	}

	const Value &reg(size_t idx) const
	{
		ASSERT(idx < registers().size())
		return registers()[idx];
	}

	Registers &registers()
	{
		ASSERT(!m_stack.empty())
		return m_stack.top().registers;
	}
	const Registers &registers() const
	{
		ASSERT(!m_stack.empty())
		return m_stack.top().registers;
	}

	Heap &heap() { return m_heap; }

	Interpreter &interpreter();
	const Interpreter &interpreter() const;

	void set_instruction_pointer(InstructionVector::const_iterator pos)
	{
		m_instruction_pointer = pos;
	}

	const InstructionVector::const_iterator &instruction_pointer() const
	{
		return m_instruction_pointer;
	}

	void clear();

	void dump() const;

	int call(const std::shared_ptr<Function> &, size_t frame_size);
	void ret();

	void shutdown_interpreter(Interpreter &);

	const std::unique_ptr<InterpreterSession> &interpreter_session() const
	{
		return m_interpreter_session;
	}

  private:
	VirtualMachine();

	int execute_internal(std::shared_ptr<Program> program);

	void show_current_instruction(size_t index, size_t window) const;

	void push_frame(size_t frame_size)
	{
		if (m_stack.empty()) {
			// the stack of main doesn't need a return address, since once it is popped
			// we shut down and there is nothing left to do
			m_stack.push(StackFrame{ frame_size, InstructionVector::const_iterator{}, this });
		} else {
			// return address is the instruction after the current instruction
			const auto return_address = m_instruction_pointer;
			m_stack.push(StackFrame{ frame_size, return_address, this });
		}
	}

	void pop_frame()
	{
		if (m_stack.size() > 1) {
			auto return_value = m_stack.top().registers[0];
			ASSERT((*m_stack.top().return_address).get());
			m_instruction_pointer = m_stack.top().return_address;
			m_stack.pop();
			m_stack.top().registers[0] = std::move(return_value);
		} else {
			// FIXME: this is an ugly way to keep the state of the interpreter
			// m_stack.pop();
		}
	}
};
