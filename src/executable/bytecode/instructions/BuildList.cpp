#include "BuildList.hpp"
#include "runtime/PyList.hpp"

using namespace py;

void BuildList::execute(VirtualMachine &vm, Interpreter &) const
{
	std::vector<Value> elements;
	elements.reserve(m_size);
	if (m_size > 0) {
		auto *end = vm.stack_pointer() + m_stack_offset + m_size;
		for (auto *sp = vm.stack_pointer() + m_stack_offset; sp < end; ++sp) {
			elements.push_back(*sp);
		}
	}
	vm.reg(m_dst) = PyList::create(elements);
};
