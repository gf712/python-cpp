#include "JumpIfNotExceptionMatch.hpp"
#include "executable/Label.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyType.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/types/builtin.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> JumpIfNotExceptionMatch::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(m_offset.has_value());
	const auto &exception_type = vm.reg(m_exception_type_reg);
	ASSERT(std::holds_alternative<PyObject *>(exception_type));
	auto *exception_type_obj = std::get<PyObject *>(exception_type);

	// there has to be at least one active exception in the current frame
	if (!interpreter.execution_frame()->exception_info().has_value()) { TODO(); }

	if (auto *type = as<PyType>(exception_type_obj)) {
		if (!interpreter.execution_frame()->exception_info()->exception->type()->issubclass(type)) {
			// skip exception handler body
			vm.set_instruction_pointer(vm.instruction_pointer() + *m_offset);
		}
	} else if (auto *types = as<PyTuple>(exception_type_obj)) {
		bool matches_any_exception = false;
		for (const auto &type : types->elements()) {
			auto obj = PyObject::from(type);
			if (obj.is_err()) { return Err(obj.unwrap_err()); }
			auto *t = as<PyType>(obj.unwrap());
			if (!t || !t->issubclass(types::base_exception())) {
				return Err(type_error(
					"catching classes that do not inherit from BaseException is not allowed"));
			}
			if (interpreter.execution_frame()->exception_info()->exception->type()->issubclass(t)) {
				matches_any_exception = true;
				break;
			}
		}
		if (!matches_any_exception) {
			// skip exception handler body
			vm.set_instruction_pointer(vm.instruction_pointer() + *m_offset);
		}
	} else {
		return Err(
			type_error("catching classes that do not inherit from BaseException is not allowed"));
	}

	return Ok(py_none());
}

void JumpIfNotExceptionMatch::relocate(size_t instruction_idx)
{
	m_offset = m_label->position() - instruction_idx - 1;
}

std::vector<uint8_t> JumpIfNotExceptionMatch::serialize() const
{
	ASSERT(m_offset.has_value());

	std::vector<uint8_t> result{
		JUMP_IF_NOT_EXCEPTION_MATCH,
		m_exception_type_reg,
	};
	::serialize(*m_offset, result);
	return result;
}
