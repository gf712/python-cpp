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
	size_t locals_count,
	size_t stack_size,
	std::string function_name,
	InstructionVector instructions,
	std::shared_ptr<Program> program)
	: Function(register_count,
		locals_count,
		stack_size,
		function_name,
		FunctionExecutionBackend::BYTECODE,
		std::move(program)),
	  m_instructions(std::move(instructions))
{}

std::string Bytecode::to_string() const
{
	std::ostringstream os;
	for (const auto &ins : m_instructions) {
		os << fmt::format("    {} {}", (void *)ins.get(), ins->to_string()) << '\n';
	}

	return os.str();
}

std::vector<uint8_t> Bytecode::serialize() const
{
	std::vector<uint8_t> result;

	py::serialize(m_register_count, result);
	py::serialize(m_locals_count, result);
	py::serialize(m_stack_size, result);
	py::serialize(m_function_name, result);
	py::serialize(static_cast<uint8_t>(m_backend), result);

	const size_t instruction_count = m_instructions.size();
	py::serialize(instruction_count, result);

	for (const auto &ins : m_instructions) {
		std::cout << ins->to_string() << std::endl;
		auto serialized_instruction = ins->serialize();
		result.insert(result.end(), serialized_instruction.begin(), serialized_instruction.end());
	}
	return result;
}

std::unique_ptr<Bytecode> Bytecode::deserialize(std::span<const uint8_t> &buffer,
	std::shared_ptr<Program> program)
{
	const auto register_count = py::deserialize<size_t>(buffer);
	const auto locals_count = py::deserialize<size_t>(buffer);
	const auto stack_size = py::deserialize<size_t>(buffer);
	const auto function_name = py::deserialize<std::string>(buffer);
	const auto backend = static_cast<FunctionExecutionBackend>(py::deserialize<uint8_t>(buffer));
	(void)backend;

	InstructionVector instructions;
	const auto instruction_count = py::deserialize<size_t>(buffer);

	for (size_t i = 0; i < instruction_count; ++i) {
		auto instruction = ::deserialize(buffer);
		if (!instruction) {
			for (const auto &ins : instructions) { std::cout << ins->to_string() << '\n'; }
			std::abort();
		}
		instructions.push_back(std::move(instruction));
	}

	return std::make_unique<Bytecode>(
		register_count, locals_count, stack_size, function_name, std::move(instructions), std::move(program));
}

PyResult<Value> Bytecode::call(VirtualMachine &vm, Interpreter &interpreter) const
{
	// create main stack frame
	[[maybe_unused]] auto main_frame = [&vm, this]() -> std::unique_ptr<StackFrame> {
		if (vm.stack().empty()) {
			return vm.setup_call_stack(m_register_count, m_locals_count, m_stack_size);
		}
		return nullptr;
	}();

	vm.set_instruction_pointer(begin());

	return eval_loop(vm, interpreter);
}

PyResult<Value> Bytecode::call_without_setup(VirtualMachine &vm, Interpreter &interpreter) const
{
	// create main stack frame
	ASSERT(!vm.stack().empty());

	constexpr auto sentinel = decltype(vm.stack().top().get().last_instruction_pointer)();
	if (vm.stack().top().get().last_instruction_pointer == sentinel) {
		// first time calling with the stack frame, so we don't have a last instruction pointer yet
		vm.set_instruction_pointer(begin());
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

	const auto stack_depth = vm.stack().size();
	const auto initial_ip = vm.instruction_pointer();

	const auto end_instruction_it = end();
	for (; vm.instruction_pointer() != end_instruction_it;
		 vm.set_instruction_pointer(std::next(vm.instruction_pointer()))) {
		ASSERT((*vm.instruction_pointer()).get());
		const auto &current_ip = vm.instruction_pointer();
		const auto &instruction = *current_ip;
		spdlog::debug("{} {}", (void *)instruction.get(), instruction->to_string());
		auto result = instruction->execute(vm, vm.interpreter());
		// we left the current stack frame in the previous instruction
		if (vm.stack().size() != stack_depth) {
			ASSERT(result.is_ok());
			return result;
		}
		// vm.dump();
		if (result.is_err()) {
			auto *exception = result.unwrap_err();
			size_t tb_lineno = 0;
			size_t tb_lasti = std::distance(initial_ip, current_ip);
			PyTraceback *tb_next = exception->traceback();
			auto traceback =
				PyTraceback::create(interpreter.execution_frame(), tb_lasti, tb_lineno, tb_next);
			ASSERT(traceback.is_ok());
			exception->set_traceback(traceback.unwrap());

			interpreter.raise_exception(exception);

			ASSERT(vm.state().cleanup.size() > 0);
			if (!vm.state().cleanup.top()) {
				ASSERT(vm.state().cleanup.size() == 1);
				// when a function returns without handling the exception do not copy the value
				// to the callers the return register
				vm.pop_frame(false);
				return result;
			} else {
				auto [exit_cleanup_type, exit_ins] = *vm.state().cleanup.top();
				vm.leave_cleanup_handling();
				vm.set_instruction_pointer(exit_ins);
			}
		} else {
			value = result.unwrap();
		}
	}

	ASSERT(value.has_value());
	return Ok(*value);
}
