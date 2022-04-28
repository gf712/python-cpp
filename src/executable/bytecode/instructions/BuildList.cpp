#include "BuildList.hpp"
#include "runtime/PyList.hpp"

using namespace py;

py::PyResult BuildList::execute(VirtualMachine &vm, Interpreter &) const
{
	std::vector<Value> elements;
	elements.reserve(m_size);
	if (m_size > 0) {
		auto *end = vm.stack_pointer() + m_stack_offset + m_size;
		for (auto *sp = vm.stack_pointer() + m_stack_offset; sp < end; ++sp) {
			elements.push_back(*sp);
		}
	}

	auto result = PyList::create(elements);
	if (result.is_ok()) { vm.reg(m_dst) = result.unwrap(); }
	return result;
};

std::vector<uint8_t> BuildList::serialize() const
{
	ASSERT(m_size < std::numeric_limits<uint8_t>::max())
	ASSERT(m_stack_offset < std::numeric_limits<uint8_t>::max())

	return {
		BUILD_LIST,
		m_dst,
		static_cast<uint8_t>(m_size),
		static_cast<uint8_t>(m_stack_offset),
	};
}
