#include "LoadName.hpp"
#include "executable/bytecode/serialization/serialize.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/NameError.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LoadName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	if (auto result = interpreter.get_object(m_object_name); result.is_ok()) {
		vm.reg(m_destination) = result.unwrap();
		return Ok(Value{ result.unwrap() });
	} else {
		return Err(result.unwrap_err());
	}
}

std::vector<uint8_t> LoadName::serialize() const
{
	std::vector<uint8_t> result = {
		LOAD_NAME,
		m_destination,
	};

	py::serialize(m_object_name, result);

	return result;
}
