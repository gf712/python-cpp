#include "BuildDict.hpp"
#include "runtime/PyDict.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> BuildDict::execute(VirtualMachine &vm, Interpreter &) const
{
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

	return PyDict::create(map).and_then([&vm, this](PyDict *dict) {
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
