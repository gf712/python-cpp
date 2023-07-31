#include "FunctionCallEx.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> FunctionCallEx::execute(VirtualMachine &vm, Interpreter &) const
{
	auto func = vm.reg(m_function);
	ASSERT(std::get_if<PyObject *>(&func));
	auto callable_object = std::get<PyObject *>(func);

	auto args = [&]() -> PyResult<PyTuple *> {
		if (m_expand_args) {
			auto tuple = vm.reg(m_args);
			ASSERT(std::holds_alternative<PyObject *>(tuple))
			ASSERT(as<PyTuple>(std::get<PyObject *>(tuple)))
			return Ok(as<PyTuple>(std::get<PyObject *>(tuple)));
		} else {
			return PyTuple::create();
		}
	}();
	if (args.is_err()) { return Err(args.unwrap_err()); }

	auto kwargs = [&]() -> PyResult<PyDict *> {
		if (m_expand_kwargs) {
			auto dict = vm.reg(m_kwargs);
			ASSERT(std::holds_alternative<PyObject *>(dict))
			ASSERT(as<PyDict>(std::get<PyObject *>(dict)))
			return Ok(as<PyDict>(std::get<PyObject *>(dict)));
		} else {
			return PyDict::create();
		}
	}();
	if (kwargs.is_err()) { return Err(args.unwrap_err()); }

	spdlog::debug("calling function '{}' with argument expansion: args='{}', kwargs='{}'",
		callable_object->to_string(),
		args.unwrap()->to_string(),
		kwargs.unwrap()->to_string());

	return callable_object->call(args.unwrap(), kwargs.unwrap()).and_then([&vm](PyObject *result) {
		vm.reg(0) = result;
		return Ok(result);
	});
}

std::vector<uint8_t> FunctionCallEx::serialize() const
{
	return {
		FUNCTION_CALL_EX,
		m_function,
		m_args,
		m_kwargs,
		m_expand_args,
		m_expand_kwargs,
	};
}