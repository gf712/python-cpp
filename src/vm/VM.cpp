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

#include <iostream>

using namespace py;

#define DEBUG_VM 0

StackFrame::StackFrame(size_t register_count,
	size_t stack_size,
	InstructionBlock::const_iterator return_address,
	VirtualMachine *vm_)
	: registers(register_count, nullptr), locals(stack_size, nullptr),
	  return_address(return_address), vm(vm_)
{
	state = std::make_unique<State>();
	spdlog::debug("Added frame with {} registers and stack size {}. New stack count: {}",
		registers.size(),
		locals.size(),
		vm->m_stack.size());
}

StackFrame::StackFrame(StackFrame &&other)
	: registers(std::move(other.registers)), locals(std::move(other.locals)),
	  return_address(other.return_address), vm(std::exchange(other.vm, nullptr)),
	  state(std::exchange(other.state, nullptr))
{}

StackFrame::~StackFrame()
{
	if (vm) { spdlog::debug("Popping frame. New stack size: {}", vm->m_stack.size()); }
}

VirtualMachine::VirtualMachine() : m_heap(Heap::the())
{
	uintptr_t *rbp;
	asm volatile("movq %%rbp, %0" : "=r"(rbp));
	m_heap.set_start_stack_pointer(rbp);
}

void VirtualMachine::setup_call_stack(size_t register_count, size_t stack_size)
{
	push_frame(register_count, stack_size);
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
		pop_frame();
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
	size_t i = 0;
	ASSERT(registers().has_value())
	ASSERT(stack_locals().has_value())

	std::cout << "Stack: " << (void *)(stack_locals()->get().data()) << " \n";
	for (const auto &register_ : stack_locals()->get()) {
		std::visit(overloaded{ [&i](const auto &stack_value) {
								  std::ostringstream os;
								  os << stack_value;
								  std::cout << fmt::format("[{}]  {}\n", i++, os.str());
							  },
					   [&i](PyObject *obj) {
						   if (obj) {
							   std::cout << fmt::format("[{}]  {} ({})\n",
								   i++,
								   static_cast<const void *>(obj),
								   obj->to_string());
						   } else {
							   std::cout << fmt::format("[{}]  (Empty)\n", i++);
						   }
					   } },
			register_);
	}

	i = 0;
	std::cout << "Register state: " << (void *)(registers()->get().data()) << " \n";
	for (const auto &register_ : registers()->get()) {
		std::visit(overloaded{ [&i](const auto &register_value) {
								  std::ostringstream os;
								  os << register_value;
								  std::cout << fmt::format("[{}]  {}\n", i++, os.str());
							  },
					   [&i](PyObject *obj) {
						   if (obj) {
							   std::cout << fmt::format("[{}]  {} ({})\n",
								   i++,
								   static_cast<const void *>(obj),
								   obj->to_string());
						   } else {
							   std::cout << fmt::format("[{}]  (Empty)\n", i++);
						   }
					   } },
			register_);
	}
}


void VirtualMachine::clear()
{
	m_heap.reset();
	while (!m_stack.empty()) m_stack.pop();
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

void VirtualMachine::push_frame(size_t register_count, size_t stack_size)
{
	if (m_stack.empty()) {
		// the stack of main doesn't need a return address, since once it is popped
		// we shut down and there is nothing left to do
		m_stack.push(
			StackFrame{ register_count, stack_size, InstructionBlock::const_iterator{}, this });
	} else {
		// return address is the instruction after the current instruction
		const auto return_address = m_instruction_pointer;
		m_stack.push(StackFrame{ register_count, stack_size, return_address, this });
	}
	spdlog::debug("Pushing frame. New stack size: {}", m_stack.size());

	// set a new state for this stack frame
	m_state = m_stack.top().state.get();

	const auto &r = registers();
	auto &stack_objects = m_stack_objects.emplace_back();
	if (r.has_value()) {
		for (const auto &v : r->get()) { stack_objects.push_back(&v); }
	}

	const auto &l = stack_locals();
	if (l.has_value()) {
		for (const auto &v : l->get()) { stack_objects.push_back(&v); }
	}
}

void VirtualMachine::pop_frame()
{
	if (m_stack.size() > 1) {
		auto return_value = m_stack.top().registers[0];
		ASSERT((*m_stack.top().return_address).get());
		m_instruction_pointer = m_stack.top().return_address;
		m_stack.pop();
		m_stack.top().registers[0] = std::move(return_value);
		// restore stack frame state
		m_state = m_stack.top().state.get();
		m_stack_objects.pop_back();
	} else {
		// FIXME: this is an ugly way to keep the state of the interpreter
		m_stack.pop();
		m_stack_objects.pop_back();
	}
}

PyModule *import(PyString *path)
{
	(void)path;
	TODO();
	return nullptr;
}
