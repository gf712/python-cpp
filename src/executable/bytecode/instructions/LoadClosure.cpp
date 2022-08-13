#include "LoadClosure.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyCell.hpp"
#include "runtime/PyFrame.hpp"
#include "vm/VM.hpp"


using namespace py;

PyResult<Value> LoadClosure::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto result = interpreter.execution_frame()->freevars()[m_source];
	vm.reg(m_destination) = result;
	return Ok(Value{ result });
}

std::vector<uint8_t> LoadClosure::serialize() const
{
	return {
		LOAD_CLOSURE,
		m_destination,
		m_source,
	};
}