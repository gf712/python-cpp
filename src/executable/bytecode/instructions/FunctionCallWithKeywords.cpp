#include "FunctionCallWithKeywords.hpp"
#include "FunctionCall.hpp"

#include "runtime/PyDict.hpp"
#include "runtime/PyTuple.hpp"

using namespace py;

PyResult FunctionCallWithKeywords::execute(VirtualMachine &vm, Interpreter &) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<PyObject *>(&func));
	auto function_object = std::get<PyObject *>(func);

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }
	auto args_tuple = PyTuple::create(args);

	if (args_tuple.is_err()) { return args_tuple; }
	ASSERT(args_tuple.is_ok());

	PyDict::MapType map;

	ASSERT(m_kwargs.size() == m_keywords.size())

	for (size_t i = 0; i < m_kwargs.size(); ++i) {
		map.insert_or_assign(String{ m_keywords[i] }, vm.reg(m_kwargs[i]));
	}

	auto kwargs_dict = PyDict::create(map);
	if (kwargs_dict.is_err()) { return kwargs_dict; }
	ASSERT(kwargs_dict.is_ok());

	auto result =
		function_object->call(args_tuple.unwrap_as<PyTuple>(), kwargs_dict.unwrap_as<PyDict>());
	if (result.is_ok()) { vm.reg(0) = result.unwrap(); }
	return result;
}

std::vector<uint8_t> FunctionCallWithKeywords::serialize() const
{
	TODO();
	return {
		FUNCTION_CALL_WITH_KW,
		m_function_name,
	};
}