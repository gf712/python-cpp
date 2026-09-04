module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


// After the import: this header names Instruction and Register, which the
// module owns, so it cannot be parsed before py.runtime is visible.

using namespace py;

PyResult<Value> BuildDict::execute(VirtualMachine &vm, Interpreter &) const
{
	// Hashing the keys may run a Python __hash__/__eq__ and clobber r0.
	auto dict_ = [&] {
		[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;

		PyDict::MapType map;

		if (m_size > 0) {
			auto *start = vm.sp() - (m_size * 2);
			for (size_t i = 0; i < m_size; ++i) {
				const auto &key = *start;
				const auto &value = *(start + m_size);
				map.emplace(key, value);
				start = std::next(start);
			}
		}

		return PyDict::create(map);
	}();

	return dict_.and_then([&vm, this](PyDict *dict) {
		vm.reg(m_dst) = dict;
		return Ok(dict);
	});
}

std::vector<uint8_t> BuildDict::serialize() const
{
	ASSERT(m_size < std::numeric_limits<uint8_t>::max());

	return {
		BUILD_DICT,
		m_dst,
		static_cast<uint8_t>(m_size),
	};
}
