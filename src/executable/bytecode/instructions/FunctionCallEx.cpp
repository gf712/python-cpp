#include "FunctionCallEx.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTuple.hpp"

using namespace py;

void FunctionCallEx::execute(VirtualMachine &vm, Interpreter &) const
{
	auto func = vm.reg(m_function);
	ASSERT(std::get_if<PyObject *>(&func));
	auto callable_object = std::get<PyObject *>(func);

	auto *args = [&]() -> PyTuple * {
		if (m_expand_args) {
			auto tuple = vm.reg(m_args);
			ASSERT(std::holds_alternative<PyObject *>(tuple))
			ASSERT(as<PyTuple>(std::get<PyObject *>(tuple)))
			return as<PyTuple>(std::get<PyObject *>(tuple));
		} else {
			return nullptr;
		}
	}();

	auto *kwargs = [&]() -> PyDict * {
		if (m_expand_kwargs) {
			auto dict = vm.reg(m_kwargs);
			ASSERT(std::holds_alternative<PyObject *>(dict))
			ASSERT(as<PyDict>(std::get<PyObject *>(dict)))
			return as<PyDict>(std::get<PyObject *>(dict));
		} else {
			return nullptr;
		}
	}();

	vm.reg(0) = callable_object->call(args, kwargs);
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