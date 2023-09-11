#include "LoadFast.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/UnboundLocalError.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LoadFast::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto result = vm.stack_local(m_stack_index);
	if (std::holds_alternative<PyObject *>(result) && !std::get<PyObject *>(result)) {
		const auto &varname = interpreter.execution_frame()->code()->varnames()[m_stack_index];
		return Err(
			unbound_local_error("local variable '{}' referenced before assignment", varname));
	}
	vm.reg(m_destination) = result;
	return Ok(result);
}

std::vector<uint8_t> LoadFast::serialize() const
{
	return {
		LOAD_FAST,
		m_destination,
		m_stack_index,
	};
}