#include "BuildDict.hpp"
#include "runtime/PyDict.hpp"

using namespace py;

void BuildDict::execute(VirtualMachine &vm, Interpreter &) const
{
	PyDict::MapType map;
	for (size_t i = 0; i < m_keys.size(); ++i) {
		map.emplace(vm.reg(m_keys[i]), vm.reg(m_values[i]));
	}

	auto &heap = vm.heap();
	vm.reg(m_dst) = heap.allocate<PyDict>(map);
};