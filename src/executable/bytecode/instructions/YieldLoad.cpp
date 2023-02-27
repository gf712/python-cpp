#include "YieldLoad.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyGenerator.hpp"
#include "vm/VM.hpp"


using namespace py;

PyResult<Value> YieldLoad::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->generator() != nullptr);

	if (auto *generator = as<PyGenerator>(interpreter.execution_frame()->generator())) {
		ASSERT(generator->last_sent_value());
		vm.reg(m_dst) = generator->last_sent_value();
	} else {
		TODO();
	}

	return Ok(vm.reg(m_dst));
}

std::vector<uint8_t> YieldLoad::serialize() const
{
	return {
		YIELD_LOAD,
		m_dst,
	};
}
