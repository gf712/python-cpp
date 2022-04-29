#include "Bytecode.hpp"
#include "instructions/Instructions.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyTraceback.hpp"
#include "serialization/deserialize.hpp"
#include "serialization/serialize.hpp"

using namespace py;

Bytecode::Bytecode(size_t register_count,
	size_t stack_size,
	std::string function_name,
	InstructionVector &&instructions,
	std::vector<View> block_views)
	: Function(register_count, stack_size, function_name, FunctionExecutionBackend::BYTECODE),
	  m_instructions(std::move(instructions)), m_block_views(block_views)
{}

std::string Bytecode::to_string() const
{
	std::ostringstream os;
	size_t block_idx{ 0 };
	for (const auto &block : m_block_views) {
		os << "- block " << block_idx++ << ":\n";
		for (const auto &ins : block) {
			os << fmt::format("    {} {}", (void *)ins.get(), ins->to_string()) << '\n';
		}
	}

	return os.str();
}

std::vector<uint8_t> Bytecode::serialize() const
{
	std::vector<uint8_t> result;

	py::serialize(m_register_count, result);
	py::serialize(m_stack_size, result);
	py::serialize(m_function_name, result);
	py::serialize(static_cast<uint8_t>(m_backend), result);

	const size_t block_count = m_block_views.size();
	py::serialize(block_count, result);
	for (const auto &block : m_block_views) {
		const size_t block_size = block.end() - block.begin();
		py::serialize(block_size, result);

		for (const auto &ins : block) {
			auto serialized_instruction = ins->serialize();
			result.insert(
				result.end(), serialized_instruction.begin(), serialized_instruction.end());
		}
	}
	return result;
}

std::unique_ptr<Bytecode> Bytecode::deserialize(std::span<const uint8_t> &buffer)
{
	const auto register_count = py::deserialize<size_t>(buffer);
	const auto stack_size = py::deserialize<size_t>(buffer);
	const auto function_name = py::deserialize<std::string>(buffer);
	const auto backend = static_cast<FunctionExecutionBackend>(py::deserialize<uint8_t>(buffer));
	(void)backend;

	InstructionVector instructions;
	std::vector<View> block_views;
	const auto block_count = py::deserialize<size_t>(buffer);

	for (size_t i = 0, ins_index_in_block_count = 0; i < block_count; ++i) {
		const auto block_size = py::deserialize<size_t>(buffer);
		for (size_t ins_index_in_block = 0; ins_index_in_block < block_size; ++ins_index_in_block) {
			instructions.push_back(::deserialize(buffer));
		}
		InstructionVector::const_iterator start = instructions.begin() + ins_index_in_block_count;
		InstructionVector::const_iterator end = instructions.end();
		block_views.emplace_back(start, end);
		ins_index_in_block_count = instructions.size();
	}

	return std::make_unique<Bytecode>(
		register_count, stack_size, function_name, std::move(instructions), block_views);
}

py::PyResult Bytecode::call(VirtualMachine &vm, Interpreter &interpreter) const
{
	std::optional<py::Value> value;
	// create main stack frame
	if (vm.stack().empty()) { vm.setup_call_stack(m_register_count, m_stack_size); }

	const auto &begin_function_block = begin();
	const auto &end_function_block = end();

	vm.set_instruction_pointer(begin_function_block->begin());
	const auto stack_count = vm.stack().size();
	// can only initialize interpreter after creating the initial stack frame
	const auto initial_ip = vm.instruction_pointer();
	auto block_view = begin_function_block;
	size_t tb_lasti = 0;


	// IMPORTANT: this assumes you will not jump from a block to the middle of another.
	//            What you can do is leave a block (and not the function) at any time and
	//            start executing the next block from its first instruction
	for (; block_view != end_function_block;) {
		vm.set_instruction_pointer(block_view->begin());
		const auto end = block_view->end();
		for (; vm.instruction_pointer() != end;
			 vm.set_instruction_pointer(std::next(vm.instruction_pointer()))) {
			ASSERT((*vm.instruction_pointer()).get())
			const auto &instruction = *vm.instruction_pointer();
			tb_lasti++;
			spdlog::debug("{} {}", (void *)instruction.get(), instruction->to_string());
			auto result = instruction->execute(vm, vm.interpreter());
			// we left the current stack frame in the previous instruction
			if (vm.stack().size() != stack_count) { return result; }
			// vm.dump();
			if (result.is_err()) {
				auto *exception = result.unwrap_err();
				interpreter.raise_exception(result.unwrap_err());
				size_t tb_lineno = 0;
				PyTraceback *tb_next = exception->traceback();
				auto traceback = PyTraceback::create(
					interpreter.execution_frame(), tb_lasti - 1, tb_lineno, tb_next);
				ASSERT(traceback.is_ok())
				exception->set_traceback(traceback.unwrap_as<PyTraceback>());

				if (!vm.state().catch_exception) {
					// restore instruction pointer
					// vm.set_instruction_pointer(initial_ip);
					(void)initial_ip;
					vm.ret();
					return result;
				} else {
					// stash exception so that instructions such as JumpIfNotExceptionMatch can
					// check if an except block exception matches

					// vm.interpreter().execution_frame()->push_exception();
					// if (vm.interpreter().execution_frame()->exception_info().has_value()) {
					// 	TODO();
					// }
					// vm.interpreter().set_status(Interpreter::Status::OK);
					vm.state().catch_exception = false;
					break;
				}
			} else {
				value = result.unwrap();
			}
		}
		if (vm.state().jump_block_count.has_value()) {
			// does it make sense to jump beyond the last block, i.e. to end_function_block
			// meaning that we leave the function?
			ASSERT((block_view + *vm.state().jump_block_count) < end_function_block)
			block_view += *vm.state().jump_block_count;
			vm.state().jump_block_count.reset();
		} else {
			block_view++;
		}
	}

	// vm.interpreter_session()->shutdown(interpreter);

	ASSERT(value.has_value())
	return PyResult::Ok(*value);
}
