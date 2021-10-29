#include "BytecodeGenerator.hpp"
#include "VM.hpp"
#include "instructions/Instructions.hpp"
#include "instructions/ReturnValue.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"

#include <iostream>

LocalFrame::LocalFrame(size_t frame_size, VirtualMachine *vm_) : vm(vm_)
{
	vm->push_frame(frame_size);
	spdlog::debug(
		"Added frame of size {}. New stack size: {}", frame_size, vm->m_local_registers.size());
}

LocalFrame::LocalFrame(LocalFrame &&other) : vm(std::exchange(other.vm, nullptr)) {}

LocalFrame::~LocalFrame()
{
	if (vm) {
		vm->pop_frame();
		spdlog::debug("Popping frame. New stack size: {}", vm->m_local_registers.size());
	}
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


int VirtualMachine::execute_frame()
{
	auto frame_depth = m_local_registers.size();
	const auto initial_ip = m_instruction_pointer;
	while (frame_depth <= m_local_registers.size()
		   && m_instruction_pointer < m_bytecode->instructions().size() - 1) {
		// show_current_instruction(m_instruction_pointer, 5);
		// dump();
		const auto &instruction = m_bytecode->instructions()[m_instruction_pointer++];
		instruction->execute(*this, *m_interpreter);

		if (auto exception_obj = m_interpreter->execution_frame()->exception()) {
			m_interpreter->unwind();
			// restore instruction pointer
			m_instruction_pointer = initial_ip;
			return 1;
		} else if (m_interpreter->status() == Interpreter::Status::EXCEPTION) {
			// bail, an error occured
			std::cout << m_interpreter->exception_message() << '\n';
			m_instruction_pointer = initial_ip;
			return 1;
		}
	}

	return 0;
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

		if (auto exception_obj = m_interpreter->execution_frame()->exception()) {
			m_interpreter->unwind();
			// restore instruction pointer
			return 1;
		} else if (m_interpreter->status() == Interpreter::Status::EXCEPTION) {
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

PyObject *VirtualMachine::execute_statement(std::shared_ptr<Bytecode> bytecode)
{
	static bool requires_setup = true;
	if (requires_setup) {
		spdlog::debug("Setting up interpreter");
		m_interpreter->setup();
		m_instruction_pointer = 0;
		FunctionBlock main_block{ .metadata = FunctionMetaData{ .function_name = "__main__" } };
		FunctionBlocks program_blocks;
		program_blocks.emplace_back(std::move(main_block));
		m_bytecode = std::make_shared<Bytecode>(std::move(program_blocks));
		requires_setup = false;
	}

	for (auto &&ins : bytecode->instructions()) { m_bytecode->add_instructions(std::move(ins)); }

	spdlog::debug("bytecode: \n{}", m_bytecode->to_string());

	spdlog::debug("Adding {} registers", bytecode->main_local_register_count());
	m_local_registers.emplace_back(bytecode->main_local_register_count(), nullptr);
	while (m_instruction_pointer < m_bytecode->instructions().size()) {
		// show_current_instruction(m_instruction_pointer, 5);
		// dump();
		const auto &instruction = m_bytecode->instructions()[m_instruction_pointer++];
		instruction->execute(*this, *m_interpreter);

		if (auto exception_obj = m_interpreter->execution_frame()->exception()) {
			m_interpreter->unwind();
			// restore instruction pointer
			std::cout << m_interpreter->exception_message() << '\n';
			return PyString::create(m_interpreter->exception_message());
		}
		if (m_interpreter->status() == Interpreter::Status::EXCEPTION) {
			// bail, an error occured
			std::cout << m_interpreter->exception_message() << '\n';
			return PyString::create(m_interpreter->exception_message());
		}
	}
	// return return value which is located in r0
	return std::visit([](const auto &value) { return PyObject::from(value); }, reg(0));
}