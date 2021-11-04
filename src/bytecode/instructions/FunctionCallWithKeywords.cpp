#include "FunctionCall.hpp"
#include "FunctionCallWithKeywords.hpp"

#include "runtime/PyDict.hpp"
#include "runtime/PyTuple.hpp"


void FunctionCallWithKeywords::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<PyObject *>(&func));
	auto function_object = std::get<PyObject *>(func);

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }
	auto args_tuple = PyTuple::create(args);

	ASSERT(args_tuple);

	PyDict::MapType map;

	ASSERT(m_kwargs.size() == m_keywords.size())

	for (size_t i = 0; i < m_kwargs.size(); ++i) {
		map.insert_or_assign(String{ m_keywords[i] }, vm.reg(m_kwargs[i]));
	}

	auto args_dict = vm.heap().allocate<PyDict>(map);
	ASSERT(args_dict);

	::execute(vm, interpreter, function_object, args_tuple, args_dict, nullptr);
}