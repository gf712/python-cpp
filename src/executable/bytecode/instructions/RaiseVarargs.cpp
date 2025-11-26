#include "RaiseVarargs.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTraceback.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "runtime/RuntimeError.hpp"
#include "runtime/TypeError.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> RaiseVarargs::execute(VirtualMachine &vm, Interpreter &) const
{
	if (m_exception.has_value()) {
		const auto &exception = vm.reg(*m_exception);
		ASSERT(std::holds_alternative<PyObject *>(exception))

		auto *exception_obj = std::get<PyObject *>(exception);
		if (as<PyType>(exception_obj)) {
			if (auto obj = as<PyType>(exception_obj)->__call__(nullptr, nullptr); obj.is_ok()) {
				exception_obj = obj.unwrap();
			} else {
				return obj;
			}
		}
		if (!exception_obj->type()->issubclass(BaseException::class_type())) {
			return Err(type_error("exceptions must derive from BaseException"));
		}
		if (m_cause.has_value()) {
			auto cause = PyObject::from(vm.reg(*m_cause));
			if (cause.is_err()) { return Err(cause.unwrap_err()); }
			static_cast<BaseException *>(exception_obj)->set_cause(cause.unwrap());
		}
		return Err(static_cast<BaseException *>(exception_obj));
	} else {
		// reraise
		if (!vm.interpreter().execution_frame()->exception_info().has_value()) {
			return Err(runtime_error("No active exception to reraise"));
		}
		auto *exc = vm.interpreter().execution_frame()->pop_exception();
		exc->set_traceback(exc->traceback()->m_tb_next);
		return Err(exc);
	}
}

std::vector<uint8_t> RaiseVarargs::serialize() const
{
	uint8_t count =
		static_cast<uint8_t>(m_exception.has_value()) + static_cast<uint8_t>(m_cause.has_value());
	std::vector<uint8_t> result{ RAISE_VARARGS, count };

	if (m_exception.has_value()) { result.push_back(*m_exception); }
	if (m_cause.has_value()) { result.push_back(*m_cause); }

	return result;
}
