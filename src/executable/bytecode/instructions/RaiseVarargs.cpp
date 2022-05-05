#include "RaiseVarargs.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "runtime/TypeError.hpp"

using namespace py;

PyResult<Value> RaiseVarargs::execute(VirtualMachine &vm, Interpreter &) const
{
	if (m_cause.has_value()) {
		ASSERT(m_exception.has_value())
		TODO();
	} else if (m_exception.has_value()) {
		const auto &exception = vm.reg(*m_exception);
		ASSERT(std::holds_alternative<PyObject *>(exception))

		auto *exception_obj = std::get<PyObject *>(exception);
		// if exception_obj is not an object that is a subclass of BaseException and it is not a
		// type that is of subclass BaseException the exception is invalid
		if (as<PyType>(exception_obj)) {
			TODO();
		} else if (!exception_obj->type()->issubclass(BaseException::static_type())) {
			return Err(type_error("exceptions must derive from BaseException"));
		}
		return Err(static_cast<BaseException *>(exception_obj));
	} else {
		// reraise
		TODO();
	}
	TODO();
	return Err(nullptr);
}

std::vector<uint8_t> RaiseVarargs::serialize() const
{
	return {
		RAISE_VARARGS,
		m_exception ? *m_exception : uint8_t{ 0 },
		m_cause ? *m_cause : uint8_t{ 0 },
	};
}