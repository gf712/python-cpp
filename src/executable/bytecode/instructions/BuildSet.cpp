module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> BuildSet::execute(VirtualMachine &vm, Interpreter &) const
{
	// Hashing the elements may run a Python __hash__/__eq__ and clobber r0.
	auto set_ = [&] {
		[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;

		PySet::SetType elements;
		elements.reserve(m_size);
		if (m_size > 0) {
			auto *start = vm.sp() - m_size;
			while (start != vm.sp()) {
				elements.insert(*start);
				start = std::next(start);
			}
		}
		return PySet::create(elements);
	}();

	return set_.and_then([&vm, this](PySet *set) {
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
