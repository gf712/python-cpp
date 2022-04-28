#include "LoadClosure.hpp"
#include "runtime/PyCell.hpp"


using namespace py;

PyResult LoadClosure::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	// size_t freevars_stack_offset =
	// 	vm.stack_locals()->get().size() - interpreter.execution_frame()->freevars().size();
	// const auto &value = vm.stack_local(freevars_stack_offset + m_source);
	auto result = interpreter.execution_frame()->freevars()[m_source];
	vm.reg(m_destination) = result;
	return PyResult::Ok(result);
}

std::vector<uint8_t> LoadClosure::serialize() const
{
	TODO();
	return {
		LOAD_CLOSURE,
		m_destination,
		m_source,
	};
}