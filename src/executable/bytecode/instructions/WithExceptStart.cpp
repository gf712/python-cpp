#include "WithExceptStart.hpp"

#include "executable/bytecode/serialization/serialize.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyTraceback.hpp"
#include "runtime/PyType.hpp"

using namespace py;

PyResult<Value> WithExceptStart::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
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
		vm.reg(m_result) = result.unwrap();
	} else {
		vm.reg(m_result) = py_false();
	}

	return Ok(vm.reg(m_result));
}

std::vector<uint8_t> WithExceptStart::serialize() const { TODO(); }
