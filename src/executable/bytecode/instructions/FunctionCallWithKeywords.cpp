#include "FunctionCallWithKeywords.hpp"
#include "FunctionCall.hpp"

#include "interpreter/Interpreter.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> FunctionCallWithKeywords::execute(VirtualMachine &vm,
	Interpreter &interpreter) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<PyObject *>(&func));
	auto function_object = std::get<PyObject *>(func);

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }
	auto args_tuple = PyTuple::create(args);

	if (args_tuple.is_err()) { return Err(args_tuple.unwrap_err()); }
	ASSERT(args_tuple.is_ok());

	PyDict::MapType map;

	ASSERT(m_kwargs.size() == m_keywords.size());

	for (size_t i = 0; i < m_kwargs.size(); ++i) {
		const auto &keyword = interpreter.execution_frame()->names(m_keywords[i]);
		map.insert_or_assign(String{ keyword }, vm.reg(m_kwargs[i]));
	}

	auto kwargs_dict = PyDict::create(map);
	if (kwargs_dict.is_err()) { return Err(kwargs_dict.unwrap_err()); }
	ASSERT(kwargs_dict.is_ok());

	spdlog::debug("calling function with arguments: \'{}\'", function_object->to_string());

	if (auto result = function_object->call(args_tuple.unwrap(), kwargs_dict.unwrap());
		result.is_ok()) {
		vm.reg(0) = result.unwrap();
		return Ok(Value{ result.unwrap() });
	} else {
		return Err(result.unwrap_err());
	}
}

std::vector<uint8_t> FunctionCallWithKeywords::serialize() const
{
	std::vector<uint8_t> result{
		FUNCTION_CALL_WITH_KW,
		m_function_name,
	};

	py::serialize(m_args, result);
	py::serialize(m_kwargs, result);
	py::serialize(m_keywords, result);

	return result;
}