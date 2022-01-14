#include "RaiseVarargs.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTuple.hpp"

using namespace py;

void RaiseVarargs::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	if (m_cause.has_value()) {
		ASSERT(m_exception.has_value())
		TODO();
	} else if (m_exception.has_value()) {
		const auto &exception = vm.reg(*m_exception);
		ASSERT(std::holds_alternative<PyObject *>(exception))
		interpreter.raise_exception(std::get<PyObject *>(exception));
	} else {
		TODO();
	}
}