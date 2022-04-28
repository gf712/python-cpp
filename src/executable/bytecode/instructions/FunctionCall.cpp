#include "FunctionCall.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTuple.hpp"

using namespace py;

PyResult FunctionCall::execute(VirtualMachine &vm, Interpreter &) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<PyObject *>(&func));
	auto callable_object = std::get<PyObject *>(func);

	std::vector<Value> args;
	args.reserve(m_size);
	if (m_size > 0) {
		auto *end = vm.stack_pointer() + m_stack_offset + m_size;
		for (auto *sp = vm.stack_pointer() + m_stack_offset; sp < end; ++sp) {
			args.push_back(*sp);
		}
	}

	auto args_tuple = PyTuple::create(args);
	if (args_tuple.is_err()) { return args_tuple; }
	spdlog::debug("args_tuple: {}", (void *)args_tuple.unwrap_as<PyTuple>());

	auto result =
		callable_object->call(as<PyTuple>(std::get<PyObject *>(args_tuple.unwrap())), nullptr);
	if (result.is_ok()) { vm.reg(0) = result.unwrap(); }
	return result;
}

std::vector<uint8_t> FunctionCall::serialize() const
{
	ASSERT(m_size < std::numeric_limits<uint8_t>::max())
	ASSERT(m_stack_offset < std::numeric_limits<uint8_t>::max())

	return {
		FUNCTION_CALL,
		m_function_name,
		static_cast<uint8_t>(m_size),
		static_cast<uint8_t>(m_stack_offset),
	};
}