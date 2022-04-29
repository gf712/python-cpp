#include "ForIter.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/StopIteration.hpp"

using namespace py;

PyResult ForIter::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto iterator = vm.reg(m_src);
	interpreter.execution_frame()->set_exception_to_catch(stop_iteration(""));
	if (auto *iterable_object = std::get_if<PyObject *>(&iterator)) {
		const auto &next_value = (*iterable_object)->next();
		if (next_value.is_err()) {
			auto *last_exception = next_value.unwrap_err();
			interpreter.raise_exception(last_exception);
			if (!interpreter.execution_frame()->catch_exception(last_exception)) {
				// exit loop in error state and handle unwinding to interpreter
				return PyResult::Err(static_cast<BaseException *>(last_exception));
			} else {
				interpreter.execution_frame()->pop_exception();
				if (interpreter.execution_frame()->exception_info().has_value()) { TODO(); }
				interpreter.set_status(Interpreter::Status::OK);
				// FIXME: subtract one since the vm will advance the ip by one.
				//        is this always true?
				vm.set_instruction_pointer(vm.instruction_pointer() + m_exit_label->position() - 1);
			}
			return PyResult::Ok(py_none());
		}
		if (next_value.is_ok()) { vm.reg(m_dst) = next_value.unwrap(); }
		return next_value;
	} else {
		// this is probably always going to be something that went wrong
		TODO();
		return PyResult::Err(nullptr);
	}
}

void ForIter::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_exit_label->set_position(m_exit_label->position() - instruction_idx);
}

std::vector<uint8_t> ForIter::serialize() const
{
	ASSERT(m_exit_label->position() < std::numeric_limits<uint8_t>::max())

	return {
		FOR_ITER,
		m_dst,
		m_src,
		static_cast<uint8_t>(m_exit_label->position()),
	};
}