#include "LoadDeref.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyCell.hpp"
#include "runtime/PyFrame.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LoadDeref::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->freevars().size() > m_source)
	auto result = interpreter.execution_frame()->freevars()[m_source]->content();
	vm.reg(m_destination) = result;
	return Ok(result);
}

std::vector<uint8_t> LoadDeref::serialize() const
{
	return {
		LOAD_DEREF,
		m_destination,
		m_source,
	};
}