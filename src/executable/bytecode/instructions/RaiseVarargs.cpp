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

std::vector<uint8_t> RaiseVarargs::serialize() const
{
	return {
		RAISE_VARARGS,
		m_exception ? *m_exception : uint8_t{ 0 },
		m_cause ? *m_cause : uint8_t{ 0 },
	};
}