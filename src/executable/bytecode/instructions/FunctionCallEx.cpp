#include "FunctionCallEx.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTuple.hpp"

using namespace py;

PyResult FunctionCallEx::execute(VirtualMachine &vm, Interpreter &) const
{
	auto func = vm.reg(m_function);
	ASSERT(std::get_if<PyObject *>(&func));
	auto callable_object = std::get<PyObject *>(func);

	auto args = [&]() -> PyResult {
		if (m_expand_args) {
			auto tuple = vm.reg(m_args);
			ASSERT(std::holds_alternative<PyObject *>(tuple))
			ASSERT(as<PyTuple>(std::get<PyObject *>(tuple)))
			return PyResult::Ok(tuple);
		} else {
			return PyTuple::create();
		}
	}();
	if (args.is_err()) { return args; }

	auto kwargs = [&]() -> PyResult {
		if (m_expand_kwargs) {
			auto dict = vm.reg(m_kwargs);
			ASSERT(std::holds_alternative<PyObject *>(dict))
			ASSERT(as<PyDict>(std::get<PyObject *>(dict)))
			return PyResult::Ok(dict);
		} else {
			return PyDict::create();
		}
	}();
	if (kwargs.is_err()) { return args; }

	auto result = callable_object->call(args.unwrap_as<PyTuple>(), kwargs.unwrap_as<PyDict>());
	if (result.is_ok()) { vm.reg(0) = result.unwrap(); }
	return result;
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