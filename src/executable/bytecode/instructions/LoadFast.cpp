#include "LoadFast.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/UnboundLocalError.hpp"
#include "vm/VM.hpp"
#include <iterator>

using namespace py;

PyResult<Value> LoadFast::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto result = vm.stack_local(m_stack_index);
	if (std::holds_alternative<PyObject *>(result) && !std::get<PyObject *>(result)) {
		auto varname = interpreter.execution_frame()->code()->varnames().begin();
		const auto &cellvars = interpreter.execution_frame()->code()->m_cellvars;
		size_t idx = 0;
		while (idx < m_stack_index) {
			if (std::find(cellvars.begin(), cellvars.end(), *varname) != cellvars.end()) { ++idx; }
			varname = std::next(varname);
		}
		return Err(
			unbound_local_error("local variable '{}' referenced before assignment", *varname));
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