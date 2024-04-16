#include "BuildSet.hpp"
#include "runtime/PySet.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> BuildSet::execute(VirtualMachine &vm, Interpreter &) const
{
	PySet::SetType elements;
	elements.reserve(m_size);
	if (m_size > 0) {
		auto *start = vm.sp() - m_size;
		while (start != vm.sp()) {
			elements.insert(*start);
			start = std::next(start);
		}
	}
	return PySet::create(elements).and_then([&vm, this](PySet *set) {
		vm.reg(m_dst) = set;
		return Ok(set);
	});
}

std::vector<uint8_t> BuildSet::serialize() const
{
	ASSERT(m_size < std::numeric_limits<uint8_t>::max());

	return {
		BUILD_SET,
		m_dst,
		static_cast<uint8_t>(m_size),
	};
}