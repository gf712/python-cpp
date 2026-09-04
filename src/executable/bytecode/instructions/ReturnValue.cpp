module;
#include "core.hpp"
#include "executable/Label.hpp"
#include "spdlog/spdlog.h"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> ReturnValue::execute(VirtualMachine &vm, Interpreter &interpreter) const
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

	if (auto *generator = interpreter.execution_frame()->generator(); generator != nullptr) {
		ASSERT(as<PyGenerator>(generator));
		as<PyGenerator>(generator)->set_invalid_return(true);
		return Err(stop_iteration(PyObject::from(result).unwrap()));
	}

	vm.reg(0) = result;

	// tell the VM to return to the calling stack frame
	vm.ret();

	return Ok(result);
}

std::vector<uint8_t> ReturnValue::serialize() const
{
	return {
		RETURN_VALUE,
		m_source,
	};
}
