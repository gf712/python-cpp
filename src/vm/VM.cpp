#include "VM.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "interpreter/Interpreter.hpp"
#include "interpreter/InterpreterSession.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTraceback.hpp"
#include "utilities.hpp"
#include <cstddef>
#include <iterator>
#include <spdlog/spdlog.h>

using namespace py;


StackFrame::StackFrame(size_t register_count,
	size_t locals_count,
	size_t stack_size,
	std::optional<InstructionVector::const_iterator> return_address,
	VirtualMachine *vm_)
	: registers(register_count, nullptr),
	  locals(vm_->m_stack_pointer, vm_->m_stack_pointer + stack_size),
	  return_address(return_address), base_pointer(vm_->m_stack_pointer + locals_count),
	  stack_pointer(vm_->m_stack_pointer + locals_count), locals_count(locals_count), vm(vm_)
{
	state = std::make_unique<State>();
	spdlog::debug("Added frame with {} registers and stack size {}. New stack frame count: {}",
		registers.size(),
		locals.size(),
		vm->m_stack_frames.size());
	// have to wipe the new stack frame, to avoid finding PyObject pointers and GC'ing objects that
	// were valid in an already popped stack frame, but have at this point already been deallocated
	for (auto &local : locals) { local = nullptr; }
}

StackFrame::StackFrame(StackFrame &&other)
	: registers(std::move(other.registers)), locals_storage(std::move(other.locals_storage)),
	  return_address(other.return_address), base_pointer(other.base_pointer),
	  stack_pointer(other.stack_pointer), locals_count(other.locals_count),
	  vm(std::exchange(other.vm, nullptr)), state(std::exchange(other.state, nullptr))
{
	if (!locals_storage.empty()) {
		locals = std::span{ locals_storage.begin(), locals_storage.end() };
		other.locals = std::span<py::Value>{};
	} else if (!locals.empty()) {
		locals = std::move(other.locals);
	}
}

StackFrame::~StackFrame()
{
	if (vm) {
		spdlog::debug("Popping frame. New stack frame count: {}", vm->m_stack_frames.size());
	}
	for (auto &local : locals) { local = nullptr; }
}

StackFrame StackFrame::clone() const
{
	StackFrame stack_frame;
	stack_frame.registers = registers;
	stack_frame.locals_storage = std::vector<py::Value>{ locals.begin(), locals.end() };
	stack_frame.locals =
		std::span{ stack_frame.locals_storage.begin(), stack_frame.locals_storage.end() };
	stack_frame.return_address = return_address;
	stack_frame.last_instruction_pointer = last_instruction_pointer;
	stack_frame.base_pointer = base_pointer;
	stack_frame.stack_pointer = stack_pointer;
	stack_frame.locals_count = locals_count;
	stack_frame.vm = vm;
	stack_frame.state = std::make_unique<State>(*state);
	return stack_frame;
}


StackFrame &StackFrame::restore()
{
	ASSERT(vm);
	// A suspended frame (e.g. a generator paused at `yield`) holds its values
	// off-stack in locals_storage, while base_pointer/stack_pointer still point
	// at the absolute m_stack offset where the frame was originally created.
	// Resuming at that stale offset only works when the VM stack pointer happens
	// to still be at/below it; from a deeper call (next()/list()) it would write
	// into a live frame. Rebase the whole region onto the current top of stack,
	// preserving the locals/operand split (locals_count) and the operand-stack
	// depth captured at suspend, so push_frame's invariants hold.
	const size_t region_size = locals_storage.size();
	const auto operand_depth =
		std::distance<std::vector<py::Value>::const_iterator>(base_pointer, stack_pointer);
	ASSERT(operand_depth >= 0);
	const auto new_start = vm->m_stack_pointer;
	auto dst = new_start;
	for (const auto &el : locals_storage) { *dst++ = el; }
	locals = std::span<py::Value>{ new_start, new_start + region_size };
	base_pointer = new_start + locals_count;
	stack_pointer = new_start + locals_count + operand_depth;
	vm->push_frame(*this);
	return vm->stack().back();
}

void StackFrame::leave()
{
	ASSERT(vm);
	// ASSERT(vm->m_stack_frames.back() == *this);
	vm->pop_frame(true);
}

VirtualMachine::VirtualMachine()
	: m_stack(kStackSize, nullptr), m_stack_pointer(m_stack.begin()),
	  m_base_pointer(m_stack_pointer), m_heap(Heap::create())
{
	uintptr_t *rbp;
	asm volatile("movq %%rbp, %0" : "=r"(rbp));
	m_heap->set_start_stack_pointer(rbp);
}

std::unique_ptr<StackFrame>
	VirtualMachine::setup_call_stack(size_t register_count, size_t locals_count, size_t stack_size)
{
	// Guard the invariant that m_stack is never reallocated after the VM
	// is constructed. Every active StackFrame holds raw pointers/iterators
	// into this buffer (locals span, base_pointer, stack_pointer), so any
	// grow/shrink would silently dangle them.
	ASSERT(m_stack.size() == kStackSize && "m_stack capacity changed; locals spans would dangle");
	return push_frame(register_count, locals_count, stack_size);
}

void VirtualMachine::ret()
{
	if (m_state->cleanup.top().has_value()
		&& m_state->cleanup.top()->first == State::CleanupLogic::WITH_EXIT) {
		m_state->cleanup.pop();
		ret();
	} else if (m_state->cleanup.top().has_value()
			   && m_state->cleanup.top()->first == State::CleanupLogic::CATCH_EXCEPTION) {
		if (auto exc = m_interpreter->execution_frame()->exception_info(); exc.has_value()) {
			// Make sure that we don't leave the current frame in an exception state when we have
			// exception handlers
			ASSERT(exc->traceback->m_tb_frame != m_interpreter->execution_frame());
		}
		m_state->cleanup.pop();
		ret();
	} else {
		ASSERT(m_state->cleanup.size() == 1);
		pop_frame(true);
	}
}

int VirtualMachine::execute(std::shared_ptr<Program> program) { return program->execute(this); }

Interpreter &VirtualMachine::initialize_interpreter(std::shared_ptr<Program> &&program)
{
	m_interpreter = std::make_unique<Interpreter>();
	m_interpreter->setup_main_interpreter(std::static_pointer_cast<BytecodeProgram>(program));
	return *m_interpreter;
}

Interpreter &VirtualMachine::interpreter()
{
	ASSERT(has_interpreter());
	return *m_interpreter;
}

const Interpreter &VirtualMachine::interpreter() const
{
	ASSERT(has_interpreter());
	return *m_interpreter;
}


void VirtualMachine::dump() const
{
	// size_t i = 0;
	ASSERT(registers().has_value());

	std::cout << "bp: " << static_cast<const void *>(bp()) << " \n";
	std::cout << "sp: " << static_cast<const void *>(sp()) << " \n";
	for (const auto &register_ : stack_locals()) {
		std::visit(overloaded{ [](const auto &stack_value) {
								  std::ostringstream os;
								  os << stack_value;
								  std::cout << fmt::format("@{}:  {}\n",
									  static_cast<const void *>(&stack_value),
									  os.str());
							  },
					   [](const PyObject *&obj) {
						   if (obj) {
							   //    std::cout << fmt::format("[{}]  {} ({})\n",
							   // 	   i++,
							   // 	   static_cast<const void *>(obj),
							   // 	   obj->to_string());
							   std::cout << fmt::format("@{}:  {}\n",
								   static_cast<const void *>(&obj),
								   static_cast<const void *>(obj));
						   } else {
							   std::cout << fmt::format(
								   "@{}:  (Empty)\n", static_cast<const void *>(&obj));
						   }
					   } },
			register_);
	}

	std::cout << "Register state: " << (void *)(registers()->get().data()) << " \n";
	for (size_t i = 0; const auto &register_ : registers()->get()) {
		std::visit(
			overloaded{ [&i](const auto &register_value) {
						   std::ostringstream os;
						   os << register_value;
						   std::cout << fmt::format("[{}]  {}\n", i, os.str());
					   },
				[&i](PyObject *obj) {
					if (obj) {
						std::cout << fmt::format(
							"[{}]  {} ({})\n", i, static_cast<const void *>(obj), obj->to_string());
					} else {
						std::cout << fmt::format("[{}]  (Empty)\n", i);
					}
				} },
			register_);
		++i;
	}
}


void VirtualMachine::clear()
{
	m_heap->reset();
	while (!m_stack_frames.empty()) m_stack_frames.pop_back();
	// should instruction pointer be optional?
	// m_instruction_pointer = nullptr;
}

void VirtualMachine::set_cleanup(State::CleanupLogic cleanup_type,
	InstructionVector::const_iterator exit_instruction)
{
	m_state->cleanup.push(std::make_pair(cleanup_type, exit_instruction));
}

void VirtualMachine::leave_cleanup_handling()
{
	ASSERT(m_state->cleanup.size() > 1);
	m_state->cleanup.pop();
}

std::unique_ptr<StackFrame>
	VirtualMachine::push_frame(size_t register_count, size_t locals_count, size_t stack_size)
{
	auto new_frame = m_stack_frames.empty()
						 ? StackFrame::create(register_count,
							   locals_count,
							   stack_size,
							   std::optional<InstructionVector::const_iterator>{},
							   this)
						 : StackFrame::create(register_count,
							   locals_count,
							   stack_size,
							   std::optional{ m_instruction_pointer },
							   this);
	push_frame(*new_frame);

	return new_frame;
}

void VirtualMachine::push_frame(StackFrame &frame)
{
	// Validate the incoming frame's footprint BEFORE mutating any VM state:
	// frame.locals spans [sp, sp + stack_size) inside m_stack, so its end()
	// is the highest slot the frame can legitimately address. If pushing
	// this frame would extend past m_stack's tail, bail loudly here rather
	// than corrupting unrelated memory later via push()/pop().
	const auto *frame_top = frame.locals.data() + frame.locals.size();
	const auto *stack_end = m_stack.data() + m_stack.size();
	ASSERT(frame_top <= stack_end && "VM stack overflow at push_frame");
	ASSERT(std::distance(m_stack_pointer, frame.stack_pointer) >= 0);
	ASSERT(std::distance(m_base_pointer, frame.base_pointer) >= 0);

	if (!m_stack_frames.empty()) {
		// stash the current stack pointer so we can restore it later
		m_stack_frames.back().get().stack_pointer = m_stack_pointer;
	}
	m_stack_frames.push_back(frame);
	// set a new state for this stack frame
	m_state = m_stack_frames.back().get().state.get();

	const auto &r = registers();
	auto &stack_objects = m_stack_objects.emplace_back();
	if (r.has_value()) {
		for (const auto &v : r->get()) { stack_objects.push_back(&v); }
	}
	for (const auto &v : stack_locals()) { stack_objects.push_back(&v); }

	m_stack_pointer = frame.stack_pointer;
	m_base_pointer = frame.base_pointer;

	spdlog::debug("Pushing frame. New stack frame count: {}", m_stack_frames.size());
}

std::deque<std::vector<const py::Value *>> VirtualMachine::stack_objects() const
{
	return m_stack_objects;
}

void VirtualMachine::pop_frame(bool should_return_value)
{
	if (m_stack_frames.size() > 1) {
		const size_t locals_size = m_stack_frames.back().get().locals.size();
		auto return_value = m_stack_frames.back().get().registers[0];

		// Non-top-level frames are pushed with a concrete call-site
		// instruction. If we ever see nullopt here, the VM was popping
		// past the top-of-stack frame through a non-top path.
		ASSERT(m_stack_frames.back().get().return_address.has_value());
		m_instruction_pointer = *m_stack_frames.back().get().return_address;
		auto f = m_stack_frames.back();
		f.get().stack_pointer = m_stack_pointer;
		m_stack_frames.pop_back();

		// restore stack frame state
		m_state = m_stack_frames.back().get().state.get();
		m_stack_objects.pop_back();
		if (should_return_value) {
			m_stack_frames.back().get().registers[0] = std::move(return_value);
		}

		f.get().locals_storage.resize(locals_size, nullptr);
		for (size_t i = 0; i < locals_size; ++i) { f.get().locals_storage[i] = f.get().locals[i]; }
		f.get().locals = std::span{ f.get().locals_storage.begin(), f.get().locals_storage.end() };

		ASSERT(std::distance(m_stack_frames.back().get().stack_pointer, m_stack_pointer) >= 0);
		ASSERT(std::distance(m_stack_frames.back().get().base_pointer, m_base_pointer) >= 0);
		m_base_pointer = m_stack_frames.back().get().base_pointer;
		m_stack_pointer = m_stack_frames.back().get().stack_pointer;
	} else {
		m_stack_frames.pop_back();
		m_stack_objects.pop_back();
	}
}
