#include "ForIter.hpp"
#include "executable/Label.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyTraceback.hpp"
#include "runtime/PyType.hpp"
#include "runtime/StopIteration.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> ForIter::execute(VirtualMachine &vm, Interpreter &) const
{
	ASSERT(m_offset.has_value())
	ASSERT(m_body_offset.has_value())
	auto iterator = vm.reg(m_src);
	if (auto *iterable_object = std::get_if<PyObject *>(&iterator)) {
		const auto &next_value = (*iterable_object)->next();
		if (next_value.is_err()) {
			auto *last_exception = next_value.unwrap_err();
			if (last_exception->type()->issubclass(stop_iteration()->type())) {
				vm.set_instruction_pointer(vm.instruction_pointer() + *m_offset);
				return Ok(py_none());
			} else {
				return Err(next_value.unwrap_err());
			}
		}
		vm.reg(m_dst) = next_value.unwrap();
		vm.set_instruction_pointer(vm.instruction_pointer() + *m_body_offset);
		return Ok(Value{ next_value.unwrap() });
	} else {
		// this is probably always going to be something that went wrong internally
		TODO();
		return Err(nullptr);
	}
}

void ForIter::relocate(size_t instruction_idx)
{
	m_offset = m_exit_label->position() - instruction_idx - 1;
	if (m_body_label) {
		m_body_offset = m_body_label->position() - instruction_idx - 1;
	} else {
		m_body_offset = 0;
	}
}

std::vector<uint8_t> ForIter::serialize() const
{
	ASSERT(m_offset.has_value());
	ASSERT(m_body_offset.has_value());

	std::vector<uint8_t> result{
		FOR_ITER,
		m_dst,
		m_src,
	};

	::serialize(*m_offset, result);
	::serialize(*m_body_offset, result);

	return result;
}
