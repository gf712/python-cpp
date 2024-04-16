#include "BuildList.hpp"
#include "runtime/PyList.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> BuildList::execute(VirtualMachine &vm, Interpreter &) const
{
	std::vector<Value> elements;
	elements.reserve(m_size);
	return PyList::create(std::span{ vm.sp() - m_size, m_size })
		.and_then([&vm, this](PyList *list) {
			vm.reg(m_dst) = list;
			return Ok(list);
		});
}

std::vector<uint8_t> BuildList::serialize() const
{
	ASSERT(m_size < std::numeric_limits<uint8_t>::max())

	return {
		BUILD_LIST,
		m_dst,
		static_cast<uint8_t>(m_size),
	};
}
