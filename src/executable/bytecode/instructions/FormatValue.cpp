#include "FormatValue.hpp"
#include "runtime/PyString.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> FormatValue::execute(VirtualMachine &vm, Interpreter &) const
{
	auto src = vm.reg(m_src);

	return PyObject::from(src)
		.and_then([](PyObject *obj) {
			// TODO: support other conversion functions
			return obj->str();
		})
		.and_then([&vm, this](PyString *str) -> PyResult<Value> {
			vm.reg(m_dst) = str;
			return Ok(str);
		});
}

std::vector<uint8_t> FormatValue::serialize() const
{
	return {
		FORMAT_VALUE,
		m_dst,
		m_src,
	};
}
