#include "BuildTuple.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> BuildTuple::execute(VirtualMachine &vm, Interpreter &) const
{
	std::vector<Value> elements;
	elements.reserve(m_size);
	if (m_size > 0) {
		auto *start = vm.sp() - m_size;
		while (start != vm.sp()) {
			elements.push_back(*start);
			start = std::next(start);
		}
	}

	return PyTuple::create(std::move(elements)).and_then([&vm, this](PyTuple *tuple) {
		vm.reg(m_dst) = tuple;
		return Ok(tuple);
	});
}

std::vector<uint8_t> BuildTuple::serialize() const
{
	ASSERT(m_size < std::numeric_limits<uint8_t>::max());

	return {
		BUILD_TUPLE,
		m_dst,
		static_cast<uint8_t>(m_size),
	};
}