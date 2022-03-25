#include "BuildTuple.hpp"
#include "runtime/PyTuple.hpp"

using namespace py;

void BuildTuple::execute(VirtualMachine &vm, Interpreter &) const
{
	std::vector<Value> elements;
	for (const auto &src : m_srcs) { elements.push_back(vm.reg(src)); }

	vm.reg(m_dst) = PyTuple::create(elements);
};