#include "ForIter.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyTraceback.hpp"
#include "runtime/StopIteration.hpp"

using namespace py;

PyResult<Value> ForIter::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(m_offset.has_value())
	auto iterator = vm.reg(m_src);
	interpreter.execution_frame()->set_exception_to_catch(stop_iteration(""));
	if (auto *iterable_object = std::get_if<PyObject *>(&iterator)) {
		const auto &next_value = (*iterable_object)->next();
		if (next_value.is_err()) {
			auto *last_exception = next_value.unwrap_err();

			// FIXME: this shold be done somewhere more centralized and where we can easily get the
			// instruction index and line number
			size_t tb_lineno = 0;
			size_t tb_lasti = 0;
			PyTraceback *tb_next = last_exception->traceback();
			auto traceback =
				PyTraceback::create(interpreter.execution_frame(), tb_lasti, tb_lineno, tb_next);
			ASSERT(traceback.is_ok())
			last_exception->set_traceback(traceback.unwrap());

			interpreter.raise_exception(last_exception);

			if (!interpreter.execution_frame()->catch_exception(last_exception)) {
				// exit loop in error state and handle unwinding to interpreter
				return Err(static_cast<BaseException *>(last_exception));
			} else {
				interpreter.execution_frame()->pop_exception();
				if (interpreter.execution_frame()->exception_info().has_value()) { TODO(); }
				// FIXME: subtract one since the vm will advance the ip by one.
				//        is this always true?
				vm.set_instruction_pointer(vm.instruction_pointer() + *m_offset);
			}
			return Ok(Value{ py_none() });
		}
		if (next_value.is_err()) return Err(next_value.unwrap_err());
		if (next_value.is_ok()) { vm.reg(m_dst) = next_value.unwrap(); }
		return Ok(Value{ next_value.unwrap() });
	} else {
		// this is probably always going to be something that went wrong internally
		TODO();
		return Err(nullptr);
	}
}

void ForIter::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_offset = m_exit_label->position() - instruction_idx - 1;
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