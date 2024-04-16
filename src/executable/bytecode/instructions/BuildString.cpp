#include "BuildString.hpp"
#include "runtime/PyString.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> BuildString::execute(VirtualMachine &vm, Interpreter &) const
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
	std::string result_string;
	for (const auto &el : elements) {
		if (std::holds_alternative<String>(el)) {
			result_string += std::get<String>(el).s;
		} else if (std::holds_alternative<PyObject *>(el)) {
			ASSERT(as<PyString>(std::get<PyObject *>(el)));
			result_string += as<PyString>(std::get<PyObject *>(el))->value();
		} else {
			TODO();
		}
	}

	return PyString::create(std::move(result_string)).and_then([&vm, this](PyString *str) {
		vm.reg(m_dst) = str;
		return Ok(str);
	});
}

std::vector<uint8_t> BuildString::serialize() const
{
	ASSERT(m_size < std::numeric_limits<uint8_t>::max());

	return { BUILD_STRING, m_dst, static_cast<uint8_t>(m_size) };
}
