#include "StoreDeref.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyCell.hpp"
#include "runtime/PyFrame.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> StoreDeref::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->freevars().size() > m_dst)
	auto result = PyCell::create(vm.reg(m_src));
	if (result.is_err()) { return Err(result.unwrap_err()); }
	interpreter.execution_frame()->freevars()[m_dst] = result.unwrap();
	return Ok(Value{ result.unwrap() });
}

std::vector<uint8_t> StoreDeref::serialize() const
{
	return {
		STORE_DEREF,
		m_dst,
		m_src,
	};
}
