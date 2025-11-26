#include "ToBool.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/Value.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> ToBool::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &src = vm.reg(m_src);
	return [&] {
		[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
		return truthy(src, interpreter);
	}()
			   .and_then([this, &vm](bool result) -> PyResult<Value> {
				   vm.reg(m_dst) = result ? py_true() : py_false();
				   return Ok(result ? py_true() : py_false());
			   });
}

std::vector<uint8_t> ToBool::serialize() const
{
	return {
		TO_BOOL,
		m_dst,
		m_src,
	};
}
