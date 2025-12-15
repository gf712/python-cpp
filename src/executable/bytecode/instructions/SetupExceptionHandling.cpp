#include "SetupExceptionHandling.hpp"
#include "executable/Label.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> SetupExceptionHandling::execute(VirtualMachine &vm, Interpreter &) const
{
	ASSERT(m_offset.has_value());
	vm.set_cleanup(State::CleanupLogic::CATCH_EXCEPTION, vm.instruction_pointer() + *m_offset);
	return Ok(Value{ py_none() });
}

void SetupExceptionHandling::relocate(size_t instruction_idx)
{
	m_offset = m_label->position() - instruction_idx - 1;
}

std::vector<uint8_t> SetupExceptionHandling::serialize() const
{
	ASSERT(m_offset.has_value());

	std::vector<uint8_t> result{
		SETUP_EXCEPTION_HANDLING,
	};

	::serialize(*m_offset, result);

	return result;
}
