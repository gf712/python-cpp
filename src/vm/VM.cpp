#include "VM.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "interpreter/Interpreter.hpp"
#include "interpreter/InterpreterSession.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"

#include <iostream>

#define DEBUG_VM 0

StackFrame::StackFrame(size_t frame_size,
	InstructionVector::const_iterator return_address,
	VirtualMachine *vm_)
	: registers(frame_size, nullptr), return_address(return_address), vm(vm_)
{
	spdlog::debug("Added frame of size {}. New stack size: {}", frame_size, vm->m_stack.size());
}

StackFrame::StackFrame(StackFrame &&other)
	: registers(std::move(other.registers)), return_address(other.return_address),
	  vm(std::exchange(other.vm, nullptr))
{}

StackFrame::~StackFrame()
{
	if (vm) { spdlog::debug("Popping frame. New stack size: {}", vm->m_stack.size()); }
}


VirtualMachine::VirtualMachine()
	: m_heap(Heap::the()), m_interpreter_session(std::make_unique<InterpreterSession>())
{
	uintptr_t *rbp;
	asm volatile("movq %%rbp, %0" : "=r"(rbp));
	m_heap.set_start_stack_pointer(rbp);
}


int VirtualMachine::call(const std::shared_ptr<Function> &function, size_t frame_size)
{
	ASSERT(function->backend() == FunctionExecutionBackend::BYTECODE)
	const auto func_ip = std::static_pointer_cast<Bytecode>(function)->begin();
	int result = EXIT_SUCCESS;

	push_frame(frame_size);
	const auto stack_size = m_stack.size();
	for (m_instruction_pointer = func_ip;; ++m_instruction_pointer) {
		const auto &instruction = *m_instruction_pointer;
		// spdlog::info(instruction->to_string());
		instruction->execute(*this, interpreter());

		// we left the current stack frame in the previous instruction
		if (m_stack.size() != stack_size) { break; }
		// dump();
		if (interpreter().execution_frame()->exception()) {
			interpreter().unwind();
			break;
		} else if (interpreter().status() == Interpreter::Status::EXCEPTION) {
			// bail, an error occured
			std::cout << interpreter().exception_message() << '\n';
			break;
		}
	}

	return result;
}

void VirtualMachine::ret() { pop_frame(); }

int VirtualMachine::execute(std::shared_ptr<Program> program)
{
	const size_t frame_size = program->main_stack_size();
	// push frame BEFORE setting the ip, so that we can save the return address
	push_frame(frame_size);

	m_instruction_pointer = program->begin();
	const auto end = program->end();
	const auto stack_size = m_stack.size();
	// can only initialize interpreter after creating the initial stack frame
	m_interpreter_session->start_new_interpreter(program);
	const auto initial_ip = m_instruction_pointer;
	for (; m_instruction_pointer != end; ++m_instruction_pointer) {
		ASSERT((*m_instruction_pointer).get())
		const auto &instruction = *m_instruction_pointer;
		// spdlog::info(instruction->to_string());
		instruction->execute(*this, interpreter());
		// we left the current stack frame in the previous instruction
		if (m_stack.size() != stack_size) { break; }
		// dump();
		if (interpreter().execution_frame()->exception()) {
			interpreter().unwind();
			// restore instruction pointer
			m_instruction_pointer = initial_ip;
			return EXIT_FAILURE;
		} else if (interpreter().status() == Interpreter::Status::EXCEPTION) {
			// bail, an error occured
			std::cout << interpreter().exception_message() << '\n';
			m_instruction_pointer = initial_ip;
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}

Interpreter &VirtualMachine::interpreter() { return m_interpreter_session->interpreter(); }


const Interpreter &VirtualMachine::interpreter() const
{
	return m_interpreter_session->interpreter();
}


void VirtualMachine::show_current_instruction(size_t index, size_t window) const
{
	TODO()
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
	std::cout << "Register state: " << (void *)(registers().data()) << " \n";
	for (const auto &register_ : registers()) {
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


void VirtualMachine::shutdown_interpreter(Interpreter &interpreter)
{
	m_interpreter_session->shutdown(interpreter);
}