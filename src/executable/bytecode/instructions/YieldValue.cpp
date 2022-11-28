#include "YieldValue.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyFrame.hpp"
#include "vm/VM.hpp"


using namespace py;

PyResult<Value> YieldValue::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto result = vm.reg(m_source);

	std::visit(
		overloaded{ [](const auto &val) {
					   std::ostringstream os;
					   os << val;
					   spdlog::debug("Return value: {}", os.str());
				   },
			[](const PyObject *val) { spdlog::debug("Return value: {}", val->to_string()); } },
		result);

	ASSERT(interpreter.execution_frame()->generator() != nullptr);

	vm.reg(0) = result;

	vm.pop_frame();

	return Ok(result);
}

std::vector<uint8_t> YieldValue::serialize() const
{
	return {
		YIELD_VALUE,
		m_source,
	};
}
