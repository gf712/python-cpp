#include "BuildDict.hpp"
#include "runtime/PyDict.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> BuildDict::execute(VirtualMachine &vm, Interpreter &) const
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

	auto result = PyDict::create(map);
	if (result.is_err()) return Err(result.unwrap_err());
	if (result.is_ok()) { vm.reg(m_dst) = result.unwrap(); }
	return Ok(Value{ result.unwrap() });
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
