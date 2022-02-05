#include "BytecodeProgram.hpp"
#include "Bytecode.hpp"
#include "executable/Function.hpp"
#include "executable/Mangler.hpp"
#include "interpreter/InterpreterSession.hpp"
#include "runtime/PyFunction.hpp"

#include <numeric>

BytecodeProgram::BytecodeProgram(FunctionBlocks &&func_blocks,
	std::string filename,
	std::vector<std::string> argv)
	: Program(std::move(filename), std::move(argv))
{
	std::vector<size_t> functions_instruction_count;
	functions_instruction_count.reserve(func_blocks.size());
	for (const auto &f : func_blocks) {
		functions_instruction_count.push_back(std::transform_reduce(
			f.blocks.begin(), f.blocks.end(), 0u, std::plus<size_t>{}, [](const auto &ins) {
				return ins.size();
			}));
	}

	const auto instruction_count =
		std::accumulate(functions_instruction_count.begin(), functions_instruction_count.end(), 0u);
	// have to reserve instruction vector to avoid relocations
	// since the iterators depend on the vector memory layout
	m_instructions.reserve(instruction_count);

	auto &main_func = func_blocks.front();

	std::vector<View> main_blocks;
	main_blocks.reserve(main_func.blocks.size());

	for (size_t start_idx = 0; auto &block : main_func.blocks) {
		// ASSERT(!block.empty())
		if (block.empty()) { continue; }
		for (auto &ins : block) { m_instructions.push_back(std::move(ins)); }
		InstructionVector::const_iterator start = m_instructions.cbegin() + start_idx;
		InstructionVector::const_iterator end = m_instructions.end();
		main_blocks.emplace_back(start, end);
		start_idx = m_instructions.size();
	}

	m_main_function = std::make_shared<Bytecode>(
		main_func.metadata.register_count, main_func.metadata.function_name, main_blocks);

	for (size_t i = 1; i < func_blocks.size(); ++i) {
		auto &func = *std::next(func_blocks.begin(), i);
		std::vector<View> func_blocks_view;
		func_blocks_view.reserve(func.blocks.size());
		for (size_t start_idx = m_instructions.size(); auto &block : func.blocks) {
			// ASSERT(!block.empty())
			if (block.empty()) { continue; }
			for (auto &ins : block) { m_instructions.push_back(std::move(ins)); }
			InstructionVector::const_iterator start = m_instructions.cbegin() + start_idx;
			InstructionVector::const_iterator end = m_instructions.end();
			func_blocks_view.emplace_back(start, end);
			start_idx = m_instructions.size();
		}

		auto bytecode = std::make_shared<Bytecode>(
			func.metadata.register_count, func.metadata.function_name, func_blocks_view);

		m_functions.emplace_back(std::move(bytecode));
	}
}

size_t BytecodeProgram::main_stack_size() const { return m_main_function->registers_needed(); }

std::string BytecodeProgram::to_string() const
{
	std::stringstream ss;
	for (const auto &func : m_functions) {
		ss << func->function_name() << ":\n";
		ss << func->to_string() << '\n';
	}

	ss << "main:\n";
	ss << m_main_function->to_string() << '\n';
	return ss.str();
}

int BytecodeProgram::execute(VirtualMachine *vm)
{
	const size_t frame_size = main_stack_size();
	// push frame BEFORE setting the ip, so that we can save the return address
	vm->push_frame(frame_size);

	const auto &begin_function_block = begin();
	const auto &end_function_block = end();

	vm->set_instruction_pointer(begin_function_block->begin());
	const auto stack_size = vm->stack().size();
	// can only initialize interpreter after creating the initial stack frame
	vm->interpreter_session()->start_new_interpreter(*this);
	const auto initial_ip = vm->instruction_pointer();
	auto block_view = begin_function_block;

	// IMPORTANT: this assumes you will not jump from a block to the middle of another.
	//            What you can do is leave a block (and not the function) at any time and
	//            start executing the next block from its first instruction
	for (; block_view != end_function_block;) {
		vm->set_instruction_pointer(block_view->begin());
		const auto end = block_view->end();
		for (; vm->instruction_pointer() != end;
			 vm->set_instruction_pointer(std::next(vm->instruction_pointer()))) {
			ASSERT((*vm->instruction_pointer()).get())
			const auto &instruction = *vm->instruction_pointer();
			// spdlog::info("{} {}", (void *)instruction.get(), instruction->to_string());
			instruction->execute(*vm, vm->interpreter());
			// we left the current stack frame in the previous instruction
			if (vm->stack().size() != stack_size) { break; }
			// vm->dump();
			if (vm->interpreter().execution_frame()->exception_info().has_value()) {
				if (!vm->state().catch_exception) {
					vm->interpreter().unwind();
					// restore instruction pointer
					vm->set_instruction_pointer(initial_ip);
					return EXIT_FAILURE;
				} else {
					vm->interpreter().execution_frame()->stash_exception();
					vm->interpreter().set_status(Interpreter::Status::OK);
					vm->state().catch_exception = false;
					break;
				}
			} else if (vm->interpreter().status() == Interpreter::Status::EXCEPTION) {
				TODO();
			}
		}
		if (vm->state().jump_block_count.has_value()) {
			// does it make sense to jump beyond the last block, i.e. to end_function_block
			// meaning that we leave the function?
			ASSERT((block_view + *vm->state().jump_block_count) < end_function_block)
			block_view += *vm->state().jump_block_count;
			vm->state().jump_block_count.reset();
		} else {
			block_view++;
		}
	}

	return EXIT_SUCCESS;
}

py::PyObject *BytecodeProgram::as_pyfunction(const std::string &function_name,
	const std::vector<std::string> &argnames,
	const std::vector<py::Value> &default_values,
	const std::vector<py::Value> &kw_default_values,
	size_t positional_args_count,
	size_t kwonly_args_count,
	const CodeFlags &flags) const
{
	for (const auto &backend : m_backends) {
		if (auto *f = backend->as_pyfunction(function_name,
				argnames,
				default_values,
				kw_default_values,
				positional_args_count,
				kwonly_args_count,
				flags)) {
			return f;
		}
	}
	if (auto it = std::find_if(m_functions.begin(),
			m_functions.end(),
			[&function_name](const auto &f) { return f->function_name() == function_name; });
		it != m_functions.end()) {
		auto function = *it;
		auto *code = VirtualMachine::the().heap().allocate<py::PyCode>(function,
			argnames,
			default_values,
			kw_default_values,
			positional_args_count,
			kwonly_args_count,
			flags,
			VirtualMachine::the().interpreter().module());

		const auto &demangled_name = Mangler::default_mangler().function_demangle(function_name);
		return VirtualMachine::the().heap().allocate<py::PyFunction>(
			demangled_name, code, VirtualMachine::the().interpreter().execution_frame()->globals());
	}
	return nullptr;
}

void BytecodeProgram::add_backend(std::shared_ptr<Program> other)
{
	m_backends.push_back(std::move(other));
}

std::string FunctionBlock::to_string() const
{
	std::ostringstream os;
	os << "Function name: " << metadata.function_name << '\n';
	size_t block_idx{ 0 };
	for (const auto &block : blocks) {
		os << "  block " << block_idx++ << '\n';
		for (const auto &ins : block) { os << "    " << ins->to_string() << '\n'; }
	}
	return os.str();
}