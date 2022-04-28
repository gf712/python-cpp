#include "StoreDeref.hpp"
#include "runtime/PyCell.hpp"

using namespace py;

PyResult StoreDeref::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->freevars().size() > m_dst)
	auto result = PyCell::create(vm.reg(m_src));
	if (result.is_err()) { return result; }
	interpreter.execution_frame()->freevars()[m_dst] = result.unwrap_as<PyCell>();
	return result;
}

std::vector<uint8_t> StoreDeref::serialize() const
{
	return {
		STORE_DEREF,
		m_dst,
		m_src,
	};
}
