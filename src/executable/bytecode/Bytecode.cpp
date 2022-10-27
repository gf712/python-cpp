#include "Bytecode.hpp"
#include "instructions/Instructions.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyTraceback.hpp"
#include "serialization/deserialize.hpp"
#include "serialization/serialize.hpp"

using namespace py;

Bytecode::Bytecode(size_t register_count,
	size_t stack_size,
	std::string function_name,
	InstructionVector &&instructions,
	std::vector<View> block_views,
	std::shared_ptr<Program> program)
	: Function(register_count,
		stack_size,
		function_name,
		FunctionExecutionBackend::BYTECODE,
		std::move(program)),
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
			std::cout << ins->to_string() << std::endl;
			auto serialized_instruction = ins->serialize();
			result.insert(
				result.end(), serialized_instruction.begin(), serialized_instruction.end());
		}
	}
	return result;
}

std::unique_ptr<Bytecode> Bytecode::deserialize(std::span<const uint8_t> &buffer,
	std::shared_ptr<Program> program)
{
	const auto register_count = py::deserialize<size_t>(buffer);
	const auto stack_size = py::deserialize<size_t>(buffer);
	const auto function_name = py::deserialize<std::string>(buffer);
	const auto backend = static_cast<FunctionExecutionBackend>(py::deserialize<uint8_t>(buffer));
	(void)backend;

	InstructionVector instructions;
	std::vector<View> block_views;
	std::vector<std::pair<size_t, size_t>> block_bounds;
	const auto block_count = py::deserialize<size_t>(buffer);

	for (size_t i = 0, ins_index_in_block_count = 0; i < block_count; ++i) {
		const auto block_size = py::deserialize<size_t>(buffer);
		for (size_t ins_index_in_block = 0; ins_index_in_block < block_size; ++ins_index_in_block) {
			instructions.push_back(::deserialize(buffer));
		}
		block_bounds.emplace_back(ins_index_in_block_count, ins_index_in_block_count + block_size);
		ins_index_in_block_count = instructions.size();
	}

	for (const auto &block_bound : block_bounds) {
		InstructionVector::const_iterator start = instructions.begin() + block_bound.first;
		InstructionVector::const_iterator end = instructions.begin() + block_bound.second;
		block_views.emplace_back(start, end);
	}

	return std::make_unique<Bytecode>(register_count,
		stack_size,
		function_name,
		std::move(instructions),
		block_views,
		std::move(program));
}

PyResult<Value> Bytecode::call(VirtualMachine &vm, Interpreter &interpreter) const
{
	// create main stack frame
	[[maybe_unused]] auto main_frame = [&vm, this]() -> std::unique_ptr<StackFrame> {
		if (vm.stack().empty()) { return vm.setup_call_stack(m_register_count, m_stack_size); }
		return nullptr;
	}();

	const auto &begin_function_block = begin();
	vm.set_instruction_pointer(begin_function_block->begin());

	return eval_loop(vm, interpreter);
}

PyResult<Value> Bytecode::call_without_setup(VirtualMachine &vm, Interpreter &interpreter) const
{
	// create main stack frame
	ASSERT(!vm.stack().empty());

	constexpr auto sentinel = decltype(vm.stack().top().get().last_instruction_pointer)();
	if (vm.stack().top().get().last_instruction_pointer == sentinel) {
		// first time calling with the stack frame, so we don't have a last instruction pointer yet
		const auto &begin_function_block = begin();
		vm.set_instruction_pointer(begin_function_block->begin());
	} else {
		// otherwise resume execution, by starting execution from the instruction after the last run
		// instruction
		vm.set_instruction_pointer(vm.stack().top().get().last_instruction_pointer + 1);
	}

	return eval_loop(vm, interpreter);
}

py::PyResult<py::Value> Bytecode::eval_loop(VirtualMachine &vm, Interpreter &interpreter) const
{
	std::optional<Value> value;

	auto find_block = [this, &vm]() {
		std::optional<std::vector<View>::const_iterator> block_view_;
		for (size_t idx = 0; const auto &block : m_block_views) {
			const auto &begin = block.begin();
			const auto &end = block.end();
			if ((vm.instruction_pointer() >= begin) && (vm.instruction_pointer() <= (end - 1))) {
				block_view_ = this->begin() + idx;
				break;
			}
			idx++;
		}
		ASSERT(block_view_.has_value())
		return *block_view_;
	};

	auto block_view = find_block();
	const auto &end_function_block = end();

	const auto stack_depth = vm.stack().size();
	const auto initial_ip = vm.instruction_pointer();

	bool requires_block_jump = true;

	for (; block_view != end_function_block;) {
		const auto begin = block_view->begin();
		const auto end = block_view->end();
		spdlog::debug("begin={} end={}",
			static_cast<void *>(begin->get()),
			static_cast<void *>((end - 1)->get()));
		if (!requires_block_jump) vm.set_instruction_pointer(begin);
		requires_block_jump = false;
		for (; vm.instruction_pointer() != end;
			 vm.set_instruction_pointer(std::next(vm.instruction_pointer()))) {
			ASSERT((*vm.instruction_pointer()).get())
			const auto &current_ip = vm.instruction_pointer();
			const auto &instruction = *current_ip;
			spdlog::debug("{} {}", (void *)instruction.get(), instruction->to_string());
			auto result = instruction->execute(vm, vm.interpreter());
			// we left the current stack frame in the previous instruction
			if (vm.stack().size() != stack_depth) {
				ASSERT(result.is_ok())
				return result;
			}
			// vm.dump();
			if (result.is_err()) {
				auto *exception = result.unwrap_err();
				size_t tb_lineno = 0;
				size_t tb_lasti = std::distance(initial_ip, current_ip);
				PyTraceback *tb_next = exception->traceback();
				auto traceback = PyTraceback::create(
					interpreter.execution_frame(), tb_lasti, tb_lineno, tb_next);
				ASSERT(traceback.is_ok())
				exception->set_traceback(traceback.unwrap());

				interpreter.raise_exception(exception);

				ASSERT(vm.state().cleanup.size() > 0);
				if (!vm.state().cleanup.top()) {
					vm.ret();
					return result;
				} else {
					auto [exit_cleanup_type, exit_ins] = *vm.state().cleanup.top();
					vm.leave_cleanup_handling();
					vm.set_instruction_pointer(exit_ins);
				}
			} else {
				value = result.unwrap();
			}

			if ((vm.instruction_pointer() < begin) || (vm.instruction_pointer() > (end - 1))) {
				requires_block_jump = true;
				break;
			}
		}
		if (!requires_block_jump)
			block_view++;
		else {
			vm.set_instruction_pointer(vm.instruction_pointer() + 1);
			block_view = find_block();
		}
	}

	ASSERT(value.has_value())
	return Ok(*value);
}
