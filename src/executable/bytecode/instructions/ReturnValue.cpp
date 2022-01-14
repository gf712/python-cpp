#include "ReturnValue.hpp"

using namespace py;

void ReturnValue::execute(VirtualMachine &vm, Interpreter &) const
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

	vm.reg(0) = result;

	// tell the VM to return to the calling stack frame
	vm.ret();
}
