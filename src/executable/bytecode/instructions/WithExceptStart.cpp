#include "WithExceptStart.hpp"

#include "executable/bytecode/serialization/serialize.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyTraceback.hpp"
#include "runtime/PyType.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> WithExceptStart::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->exception_info().has_value());
	const auto &exit_method = vm.reg(m_exit_method);

	if (!std::holds_alternative<PyObject *>(exit_method)) { TODO(); }

	auto *exit_method_obj = std::get<PyObject *>(exit_method);
	ASSERT(exit_method_obj)

	auto args_tuple = [&]() {
		if (auto exc = interpreter.execution_frame()->exception_info()) {
			return PyTuple::create(static_cast<PyObject *>(exc->exception_type),
				static_cast<PyObject *>(exc->exception),
				exc->traceback ? static_cast<PyObject *>(exc->traceback) : py_none());
		} else {
			return PyTuple::create(py_none(), py_none(), py_none());
		}
	}();

	if (args_tuple.is_err()) return Err(args_tuple.unwrap_err());

	auto result = exit_method_obj->call(args_tuple.unwrap(), nullptr);

	if (result.is_ok()) {
		if (auto r = truthy(result.unwrap(), interpreter); r.is_ok()) {
			if (!r.unwrap()) {
				const auto active_exception =
					interpreter.execution_frame()->exception_info().has_value();
				vm.reg(m_result) = active_exception ? py_false() : py_true();
			} else {
				vm.reg(m_result) = py_true();
			}
		}
	} else {
		vm.reg(m_result) = py_false();
	}

	return Ok(vm.reg(m_result));
}

std::vector<uint8_t> WithExceptStart::serialize() const
{
	return {
		WITH_EXCEPT_START,
		m_result,
		m_exit_method,
	};
}
