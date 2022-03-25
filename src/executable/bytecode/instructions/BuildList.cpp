#include "BuildList.hpp"
#include "runtime/PyList.hpp"

using namespace py;

void BuildList::execute(VirtualMachine &vm, Interpreter &) const
{
	std::vector<Value> elements;
	for (const auto &src : m_srcs) { elements.push_back(vm.reg(src)); }

	vm.reg(m_dst) = PyList::create(elements);
};
