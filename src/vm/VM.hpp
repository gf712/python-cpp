#pragma once

#include "executable/FunctionBlock.hpp"
#include "forward.hpp"
#include "memory/Heap.hpp"
#include "runtime/Value.hpp"
#include "utilities.hpp"

#include <deque>
#include <functional>
#include <span>
#include <stack>
#include <vector>

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
		size_t locals_count,
		size_t stack_size,
		std::optional<InstructionVector::const_iterator> return_address,
		VirtualMachine *);

	StackFrame(StackFrame &&);

  public:
	template<typename... Args> static std::unique_ptr<StackFrame> create(Args &&...args)
	{
		return std::unique_ptr<StackFrame>(new StackFrame{ std::forward<Args>(args)... });
	}

	Registers registers;
	std::vector<py::Value> locals_storage;
	std::span<py::Value> locals;
	// nullopt for the top-level frame: there is no instruction to resume at.
	// For nested frames this holds the call-site instruction whose execution
	// triggered the push; the VM resumes at that iterator on pop, after
	// which the eval loop's `std::next` advances past it.
	std::optional<InstructionVector::const_iterator> return_address;
	InstructionVector::const_iterator last_instruction_pointer;
	std::vector<py::Value>::const_iterator base_pointer;
	std::vector<py::Value>::iterator stack_pointer;
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
	// Fixed-capacity value stack. Every StackFrame::locals is a std::span
	// pointing directly into this buffer, so the storage MUST NOT be
	// resized or reallocated for the lifetime of the VM — doing so would
	// dangle every existing frame's locals span and the m_stack_pointer
	// iterator. If the stack ever needs to grow, switch StackFrame::locals
	// to an (offset, size) pair against m_stack first.
	static constexpr size_t kStackSize = 10'000;
	std::vector<py::Value> m_stack;
	// m_stack_frames is a vector rather than a stack so callers can iterate
	// over the frame chain (used by VirtualMachine::dump and the GC root
	// scan). LIFO access is via back()/push_back()/pop_back().
	std::vector<std::reference_wrapper<StackFrame>> m_stack_frames;
	std::deque<std::vector<const py::Value *>> m_stack_objects;

	InstructionVector::const_iterator m_instruction_pointer;
	std::vector<py::Value>::iterator m_stack_pointer;
	std::vector<py::Value>::const_iterator m_base_pointer;
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
		ASSERT(r.has_value());
		ASSERT(idx < r->get().size());
		return r->get()[idx];
	}

	const py::Value &reg(size_t idx) const
	{
		auto r = registers();
		ASSERT(r.has_value());
		ASSERT(idx < r->get().size());
		return r->get()[idx];
	}

	py::Value &stack_local(size_t idx)
	{
		auto local = stack_locals();
		ASSERT(!local.empty());
		ASSERT(idx < local.size());
		return local[idx];
	}

	const py::Value &stack_local(size_t idx) const
	{
		auto local = stack_locals();
		ASSERT(!local.empty());
		ASSERT(idx < local.size());
		return local[idx];
	}

	std::optional<std::reference_wrapper<Registers>> registers()
	{
		if (!m_stack_frames.empty()) { return m_stack_frames.back().get().registers; }
		return {};
	}
	std::optional<std::reference_wrapper<const Registers>> registers() const
	{
		if (!m_stack_frames.empty()) { return m_stack_frames.back().get().registers; }
		return {};
	}

	std::span<py::Value> stack_locals()
	{
		if (!m_stack_frames.empty()) { return m_stack_frames.back().get().locals; }
		return {};
	}

	std::span<const py::Value> stack_locals() const
	{
		if (!m_stack_frames.empty()) { return m_stack_frames.back().get().locals; }
		return {};
	}

	const std::vector<std::reference_wrapper<StackFrame>> &stack() const { return m_stack_frames; }
	const State &state() const { return *m_state; }
	State &state() { return *m_state; }

	Heap &heap() { return *m_heap; }

	Interpreter &initialize_interpreter(std::shared_ptr<Program> &&);
	Interpreter &interpreter();
	const Interpreter &interpreter() const;
	bool has_interpreter() const { return static_cast<bool>(m_interpreter); }

	void set_instruction_pointer(InstructionVector::const_iterator pos)
	{
		m_instruction_pointer = pos;
		m_stack_frames.back().get().last_instruction_pointer = m_instruction_pointer;
	}

	const InstructionVector::const_iterator &instruction_pointer() const
	{
		return m_instruction_pointer;
	}

	const py::Value *sp() const { return &*m_stack_pointer; }
	const py::Value *bp() const { return &*m_base_pointer; }

	void clear();

	void dump() const;

	[[nodiscard]] std::unique_ptr<StackFrame>
		setup_call_stack(size_t register_count, size_t locals_count, size_t stack_size);

	void ret();
	void set_cleanup(State::CleanupLogic cleanup_type,
		InstructionVector::const_iterator exit_instruction);
	void leave_cleanup_handling();

	[[nodiscard]] std::unique_ptr<StackFrame>
		push_frame(size_t register_count, size_t locals_count, size_t stack_size);
	void push_frame(StackFrame &frame);

	void pop_frame(bool should_return_value);

	void push(py::Value value)
	{
		ASSERT(m_stack_pointer != m_stack.end() && "VM stack overflow on push");
		*m_stack_pointer++ = value;
	}

	py::Value pop()
	{
		ASSERT(m_stack_pointer != m_stack.begin() && "VM stack underflow on pop");
		return *--m_stack_pointer;
	}

	std::deque<std::vector<const py::Value *>> stack_objects() const;

  private:
	VirtualMachine();

	int execute_internal(std::shared_ptr<Program> program);
};

static_assert(std::is_same_v<decltype(std::declval<const VirtualMachine &>().stack_locals()),
				  std::span<const py::Value>>,
	"stack_locals() const must return a span of const py::Value");
