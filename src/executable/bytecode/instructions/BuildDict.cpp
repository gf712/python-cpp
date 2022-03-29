#include "BuildDict.hpp"
#include "runtime/PyDict.hpp"

using namespace py;

void BuildDict::execute(VirtualMachine &vm, Interpreter &) const
{
	PyDict::MapType map;

	if (m_size > 0) {
		auto *end = vm.stack_pointer() + m_stack_offset + m_size;
		for (auto *sp = vm.stack_pointer() + m_stack_offset; sp < end; ++sp) {
			const auto &key = *sp;
			const auto &value = *(sp + m_size);
			map.emplace(key, value);
		}
	}

	auto &heap = vm.heap();
	vm.reg(m_dst) = heap.allocate<PyDict>(map);
};

std::vector<uint8_t> BuildDict::serialize() const
{
	ASSERT(m_size < std::numeric_limits<uint8_t>::max())
	ASSERT(m_stack_offset < std::numeric_limits<uint8_t>::max())

	return {
		BUILD_DICT,
		m_dst,
		static_cast<uint8_t>(m_size),
		static_cast<uint8_t>(m_stack_offset),
	};
}
