#include "YieldFrom.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/BaseException.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyGenerator.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/PyType.hpp"
#include "runtime/StopIteration.hpp"
#include "vm/VM.hpp"


using namespace py;

PyResult<Value> YieldFrom::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->generator() != nullptr);

	auto src = vm.reg(m_receiver);
	auto value = vm.reg(m_value);
	ASSERT(std::holds_alternative<PyObject *>(src));
	auto receiver = std::get<PyObject *>(src);
	auto v = PyObject::from(value);
	if (v.is_err()) { return v; }

	auto result = [receiver, v = v.unwrap()]() -> PyResult<Value> {
		if (auto *generator = as<PyGenerator>(receiver)) {
			return generator->send(v);
		} else if (v == py_none()) {
			return receiver->next();
		} else {
			return receiver->get_method(PyString::create("send").unwrap())
				.and_then([v](PyObject *send) {
					return send->call(PyTuple::create(v).unwrap(), nullptr);
				});
		}
	}();

	if (result.is_err()) {
		if (result.unwrap_err()->type()->issubclass(stop_iteration()->type())) {
			const auto &args = as<StopIteration>(result.unwrap_err())->args()->elements();
			result = Ok(args.empty() ? py_none() : args[0]);
			vm.reg(m_dst) = result.unwrap();
		} else {
			TODO();
		}
	} else {
		vm.reg(m_dst) = result.unwrap();
		vm.reg(0) = result.unwrap();
		vm.set_instruction_pointer(vm.instruction_pointer() - 1);
		vm.pop_frame();
	}

	return result;
}

std::vector<uint8_t> YieldFrom::serialize() const
{
	return {
		YIELD_FROM,
		m_dst,
		m_receiver,
		m_value,
	};
}
