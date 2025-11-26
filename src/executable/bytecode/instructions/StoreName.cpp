#include "StoreName.hpp"

#include "executable/bytecode/serialization/serialize.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyNone.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> StoreName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &value = vm.reg(m_source);
	[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
	return interpreter.store_object(m_object_name, value).and_then([](auto) {
		return Ok(py_none());
	});
}

std::vector<uint8_t> StoreName::serialize() const
{
	std::vector<uint8_t> result{
		STORE_NAME,
		m_source,
	};

	py::serialize(m_object_name, result);

	return result;
}
