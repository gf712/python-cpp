#include "BytecodeGenerator.hpp"
#include "VM.hpp"
#include "instructions/Instructions.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyObject.hpp"

#include <iostream>

VirtualMachine::VirtualMachine()
	: m_interpreter(std::make_unique<Interpreter>()), m_heap(Heap::the())
{}

void VirtualMachine::push_generator(std::shared_ptr<Bytecode> bytecode)
{
	ASSERT(!m_bytecode)
	m_bytecode = std::move(bytecode);
	m_registers = { m_bytecode->virtual_register_count(), nullptr };
}


int VirtualMachine::execute()
{
	// start execution main (function_id = 0)
	m_interpreter->setup();
	m_instruction_pointer = m_bytecode->start_offset();

	while (m_instruction_pointer < m_bytecode->instructions().size() - 1) {
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
	for (const auto &register_ : m_registers) {
		std::visit(overloaded{ [&i](const auto &register_value) {
								  std::ostringstream os;
								  os << register_value;
								  std::cout << fmt::format("[{}]  {}\n", i++, os.str());
							  },
					   [&i](const std::shared_ptr<PyObject> &obj) {
						   if (obj) {
							   std::cout << fmt::format("[{}]  {} ({})\n",
								   i++,
								   static_cast<const void *>(&obj),
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
	m_registers.clear();
	m_heap.reset();
	m_instruction_pointer = 0;
}

size_t VirtualMachine::function_offset(size_t func_id)
{
	ASSERT(func_id >= 1)
	func_id--;
	return m_bytecode->functions()[func_id].first;
}