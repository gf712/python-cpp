#include "SetupWith.hpp"
#include "executable/Label.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> SetupWith::execute(VirtualMachine &vm, Interpreter &) const
{
	ASSERT(m_offset.has_value())
	vm.set_cleanup(State::CleanupLogic::WITH_EXIT, vm.instruction_pointer() + *m_offset);
	return Ok(Value{ py_none() });
}

void SetupWith::relocate(size_t instruction_idx)
{
	m_offset = m_label->position() - instruction_idx - 1;
}

std::vector<uint8_t> SetupWith::serialize() const
{
	ASSERT(m_offset.has_value())

	std::vector<uint8_t> result{
		SETUP_WITH,
	};

	::serialize(*m_offset, result);

	return result;
}
