#include "ForIter.hpp"
#include "runtime/StopIteration.hpp"

using namespace py;

void ForIter::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto iterator = vm.reg(m_src);
	interpreter.execution_frame()->set_exception_to_catch(stop_iteration(""));
	if (auto *iterable_object = std::get_if<PyObject *>(&iterator)) {
		const auto &next_value = (*iterable_object)->next();
		if (interpreter.execution_frame()->exception_info().has_value()) {
			auto *last_exception = interpreter.execution_frame()->exception_info()->exception;
			if (!interpreter.execution_frame()->catch_exception(last_exception)) {
				// exit loop in error state and handle unwinding to interpreter
			} else {
				interpreter.execution_frame()->set_exception(nullptr);
				interpreter.set_status(Interpreter::Status::OK);
				// FIXME: subtract one since the vm will advance the ip by one.
				//        is this always true?
				vm.set_instruction_pointer(vm.instruction_pointer() + m_exit_label->position() - 1);
			}
			return;
		}
		vm.reg(m_dst) = next_value;
	} else {
		// this is probably always going to be something that went wrong
		TODO();
	}
}

void ForIter::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_exit_label->set_position(m_exit_label->position() - instruction_idx);
}
