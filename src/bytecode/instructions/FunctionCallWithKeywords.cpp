#include "FunctionCall.hpp"
#include "FunctionCallWithKeywords.hpp"

#include "runtime/PyDict.hpp"


void FunctionCallWithKeywords::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto func = vm.reg(m_function_name);
	ASSERT(std::get_if<std::shared_ptr<PyObject>>(&func));
	auto function_object = std::get<std::shared_ptr<PyObject>>(func);

	std::vector<Value> args;
	for (const auto &arg_register : m_args) { args.push_back(vm.reg(arg_register)); }
	auto args_tuple = vm.heap().allocate<PyTuple>(args);

	ASSERT(args_tuple);

	std::unordered_map<Value, Value, ValueHash> map;

	ASSERT(m_kwargs.size() == m_keywords.size())

	for (size_t i = 0; i < m_kwargs.size(); ++i) {
		map.insert_or_assign(String{ m_keywords[i] }, vm.reg(m_kwargs[i]));
	}

	auto args_dict = vm.heap().allocate<PyDict>(map);
	ASSERT(args_dict);

	::execute(vm, interpreter, function_object, args_tuple, args_dict);
}