#include "JumpIfNotExceptionMatch.hpp"
#include "executable/Label.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyType.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> JumpIfNotExceptionMatch::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(m_offset.has_value())
	const auto &exception_type = vm.reg(m_exception_type_reg);
	ASSERT(std::holds_alternative<PyObject *>(exception_type))
	auto *exception_type_obj = std::get<PyObject *>(exception_type);
	ASSERT(as<PyType>(exception_type_obj))

	// there has to be at least one active exception in the current frame
	if (!interpreter.execution_frame()->exception_info().has_value()) { TODO(); }

	if (!interpreter.execution_frame()->exception_info()->exception->type()->issubclass(
			as<PyType>(exception_type_obj))) {
		// skip exception handler body
		vm.set_instruction_pointer(vm.instruction_pointer() + *m_offset);
	}
	return Ok(Value{ py_none() });
}

void JumpIfNotExceptionMatch::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_offset = m_label->position() - instruction_idx - 1;
}

std::vector<uint8_t> JumpIfNotExceptionMatch::serialize() const
{
	ASSERT(m_offset.has_value())

	std::vector<uint8_t> result{
		JUMP_IF_NOT_EXCEPTION_MATCH,
		m_exception_type_reg,
	};
	::serialize(*m_offset, result);
	return result;
}