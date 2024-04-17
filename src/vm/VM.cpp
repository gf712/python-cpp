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
#include <iterator>
#include <spdlog/spdlog.h>

using namespace py;


StackFrame::StackFrame(size_t register_count,
	size_t locals_count,
	size_t stack_size,
	InstructionVector::const_iterator return_address,
	VirtualMachine *vm_)
	: registers(register_count, nullptr),
	  locals(vm_->m_stack_pointer, vm_->m_stack_pointer + stack_size),
	  return_address(return_address), base_pointer(vm_->m_stack_pointer + locals_count),
	  stack_pointer(vm_->m_stack_pointer + locals_count), vm(vm_)
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
	  stack_pointer(other.stack_pointer), vm(std::exchange(other.vm, nullptr)),
	  state(std::exchange(other.state, nullptr))
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
	stack_frame.vm = vm;
	stack_frame.state = std::make_unique<State>(*state);
	return stack_frame;
}


StackFrame &StackFrame::restore()
{
	ASSERT(vm);
	auto start = vm->m_stack_pointer;
	for (const auto &el : locals_storage) { *start++ = el; }
	vm->push_frame(*this);
	return vm->stack().top();
}

void StackFrame::leave()
{
	ASSERT(vm);
	// ASSERT(vm->m_stack_frames.top() == *this);
	vm->pop_frame(true);
}

VirtualMachine::VirtualMachine()
	: m_stack(10'000, nullptr), m_stack_pointer(m_stack.begin()), m_base_pointer(m_stack_pointer),
	  m_heap(Heap::create())
{
	uintptr_t *rbp;
	asm volatile("movq %%rbp, %0" : "=r"(rbp));
	m_heap->set_start_stack_pointer(rbp);
}

std::unique_ptr<StackFrame>
	VirtualMachine::setup_call_stack(size_t register_count, size_t locals_count, size_t stack_size)
{
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
		ASSERT(m_state->cleanup.size() == 1)
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


void VirtualMachine::show_current_instruction(size_t index, size_t window) const
{
	TODO();
	(void)index;
	(void)window;

	// size_t start = std::max(
	// 	int64_t{ 0 }, static_cast<int64_t>(index) - static_cast<int64_t>((window - 1) / 2));
	// size_t end = std::min(index + (window - 1) / 2 + 1, m_bytecode->instructions().size());

	// for (size_t i = start; i < end; ++i) {
	// 	if (i == index) {
	// 		std::cout << "->" << m_bytecode->instructions()[i]->to_string() << '\n';
	// 	} else {
	// 		std::cout << "  " << m_bytecode->instructions()[i]->to_string() << '\n';
	// 	}
	// }
	// std::cout << '\n';
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

	// i = 0;
	// std::cout << "Register state: " << (void *)(registers()->get().data()) << " \n";
	// for (const auto &register_ : registers()->get()) {
	// 	std::visit(overloaded{ [&i](const auto &register_value) {
	// 							  std::ostringstream os;
	// 							  os << register_value;
	// 							  std::cout << fmt::format("[{}]  {}\n", i++, os.str());
	// 						  },
	// 				   [&i](PyObject *obj) {
	// 					   if (obj) {
	// 						   //    std::cout << fmt::format("[{}]  {} ({})\n",
	// 						   // 	   i++,
	// 						   // 	   static_cast<const void *>(obj),
	// 						   // 	   obj->to_string());
	// 						   std::cout << fmt::format(
	// 							   "[{}]  {}\n", i++, static_cast<const void *>(obj));
	// 					   } else {
	// 						   std::cout << fmt::format("[{}]  (Empty)\n", i++);
	// 					   }
	// 				   } },
	// 		register_);
	// }
}


void VirtualMachine::clear()
{
	m_heap->reset();
	while (!m_stack_frames.empty()) m_stack_frames.pop();
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
	ASSERT(m_state->cleanup.size() > 1)
	m_state->cleanup.pop();
}

std::unique_ptr<StackFrame>
	VirtualMachine::push_frame(size_t register_count, size_t locals_count, size_t stack_size)
{
	auto new_frame =
		m_stack_frames.empty()
			? StackFrame::create(
				register_count, locals_count, stack_size, InstructionVector::const_iterator{}, this)
			: StackFrame::create(
				register_count, locals_count, stack_size, m_instruction_pointer, this);
	push_frame(*new_frame);

	return new_frame;
}

void VirtualMachine::push_frame(StackFrame &frame)
{
	if (!m_stack_frames.empty()) {
		// stash the current stack pointer so we can restore it later
		m_stack_frames.top().get().stack_pointer = m_stack_pointer;
	}
	m_stack_frames.push(frame);
	// set a new state for this stack frame
	m_state = m_stack_frames.top().get().state.get();

	const auto &r = registers();
	auto &stack_objects = m_stack_objects.emplace_back();
	if (r.has_value()) {
		for (const auto &v : r->get()) { stack_objects.push_back(&v); }
		for (const auto &v : stack_locals()) { stack_objects.push_back(&v); }
	}

	ASSERT(std::distance(m_stack_pointer, frame.stack_pointer) >= 0);
	ASSERT(std::distance(m_base_pointer, frame.base_pointer) >= 0);

	if (std::distance(m_stack.begin(), frame.stack_pointer)
		>= static_cast<int64_t>(m_stack.size())) {
		ASSERT(false && "Stack overflow!");
	}

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
		const size_t locals_size = m_stack_frames.top().get().locals.size();

		auto return_value = m_stack_frames.top().get().registers[0];
		ASSERT((*m_stack_frames.top().get().return_address).get());
		m_instruction_pointer = m_stack_frames.top().get().return_address;
		auto f = m_stack_frames.top();
		m_stack_frames.pop();
		if (should_return_value) {
			// returning a value may not be always desirable (e.g. leaving a function in an
			// exception state)
			m_stack_frames.top().get().registers[0] = std::move(return_value);
		}

		// restore stack frame state
		m_state = m_stack_frames.top().get().state.get();
		m_stack_objects.pop_back();

		f.get().locals_storage.resize(locals_size, nullptr);
		for (size_t i = 0; i < locals_size; ++i) { f.get().locals_storage[i] = f.get().locals[i]; }
		f.get().locals = std::span{ f.get().locals_storage.begin(), f.get().locals_storage.end() };

		ASSERT(std::distance(m_stack_frames.top().get().stack_pointer, m_stack_pointer) >= 0);
		ASSERT(std::distance(m_stack_frames.top().get().base_pointer, m_base_pointer) >= 0);
		m_base_pointer = m_stack_frames.top().get().base_pointer;
		m_stack_pointer = m_stack_frames.top().get().stack_pointer;
	} else {
		m_stack_frames.pop();
		m_stack_objects.pop_back();
	}
}

PyModule *import(PyString *path)
{
	(void)path;
	TODO();
	return nullptr;
}
