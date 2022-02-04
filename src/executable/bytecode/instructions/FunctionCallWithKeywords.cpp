#include "FunctionCallWithKeywords.hpp"
#include "FunctionCall.hpp"

#include "runtime/PyDict.hpp"
#include "runtime/PyTuple.hpp"


void FunctionCallWithKeywords::execute(VirtualMachine &vm, Interpreter &) const
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

	// FIXME: process kwargs
	auto kwargs_dict = vm.heap().allocate<PyDict>(map);
	ASSERT(kwargs_dict);

	if (auto *result = function_object->call(args_tuple, kwargs_dict)) { vm.reg(0) = result; }
}