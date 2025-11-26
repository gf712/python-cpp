#include "FormatValue.hpp"
#include "runtime/PyString.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> FormatValue::execute(VirtualMachine &vm, Interpreter &) const
{
	auto src = vm.reg(m_src);

	return PyObject::from(src)
		.and_then([this](PyObject *obj) {
			if (m_conversion == 0) { return obj->str(); }
			const auto conversion =
				static_cast<PyString::ReplacementField::Conversion>(m_conversion);
			switch (conversion) {
			case PyString::ReplacementField::Conversion::ASCII: {
				TODO();
				// return obj->ascii();
			} break;
			case PyString::ReplacementField::Conversion::REPR: {
				[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;

				return obj->repr();
			} break;
			case PyString::ReplacementField::Conversion::STR: {
				[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
				return obj->str();
			} break;
			}
			ASSERT_NOT_REACHED();
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
		m_conversion,
	};
}
