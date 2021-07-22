#include "BytecodeGenerator.hpp"
#include "VM.hpp"
#include "instructions/Instructions.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyObject.hpp"

#include <iostream>

LocalFrame::LocalFrame(size_t frame_size, VirtualMachine *vm_) : vm(vm_)
{
	vm->push_frame(frame_size);
}

LocalFrame::LocalFrame(LocalFrame &&other) { vm = std::exchange(other.vm, nullptr); }

LocalFrame::~LocalFrame()
{
	if (vm) vm->pop_frame();
}

VirtualMachine::VirtualMachine()
	: m_interpreter(std::make_unique<Interpreter>()), m_heap(Heap::the())
{}

void VirtualMachine::push_generator(std::shared_ptr<Bytecode> bytecode)
{
	ASSERT(!m_bytecode)
	m_bytecode = std::move(bytecode);
	m_local_registers.emplace_back(m_bytecode->main_local_register_count(), nullptr);
}


void VirtualMachine::show_current_instruction(size_t index, size_t window) const
{
	size_t start = std::max(
		int64_t{ 0 }, static_cast<int64_t>(index) - static_cast<int64_t>((window - 1) / 2));
	size_t end = std::min(index + (window - 1) / 2 + 1, m_bytecode->instructions().size());

	for (size_t i = start; i < end; ++i) {
		if (i == index) {
			std::cout << "->" << m_bytecode->instructions()[i]->to_string() << '\n';
		} else {
			std::cout << "  " << m_bytecode->instructions()[i]->to_string() << '\n';
		}
	}
	std::cout << '\n';
}

int VirtualMachine::execute()
{
	// start execution in __main__
	m_interpreter->setup();
	m_instruction_pointer = m_bytecode->start_offset();

	while (m_instruction_pointer < m_bytecode->instructions().size() - 1) {
		// show_current_instruction(m_instruction_pointer, 5);
		// dump();
		const auto &instruction = m_bytecode->instructions()[m_instruction_pointer++];
		instruction->execute(*this, *m_interpreter);

		if (m_interpreter->status() == Interpreter::Status::EXCEPTION) {
			// bail, an error occured
			std::cout << m_interpreter->exception_message() << '\n';
			return 1;
		}
	}

	return 0;
}

void VirtualMachine::dump() const
{
	size_t i = 0;
	std::cout << "Register state: \n";
	for (const auto &register_ : m_local_registers.back()) {
		std::visit(overloaded{ [&i](const auto &register_value) {
								  std::ostringstream os;
								  os << register_value;
								  std::cout << fmt::format("[{}]  {}\n", i++, os.str());
							  },
					   [&i](const std::shared_ptr<PyObject> &obj) {
						   if (obj) {
							   std::cout << fmt::format("[{}]  {} ({})\n",
								   i++,
								   static_cast<const void *>(obj.get()),
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
	m_bytecode.reset();
	m_local_registers.clear();
	m_heap.reset();
	m_instruction_pointer = 0;
}

size_t VirtualMachine::function_offset(size_t func_id)
{
	ASSERT(func_id >= 1)
	func_id--;
	return m_bytecode->functions()[func_id].offset;
}

size_t VirtualMachine::function_register_count(size_t func_id)
{
	ASSERT(func_id >= 1)
	func_id--;
	return m_bytecode->functions()[func_id].register_count;
}