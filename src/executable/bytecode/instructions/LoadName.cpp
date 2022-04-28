#include "LoadName.hpp"
#include "executable/bytecode/serialization/serialize.hpp"
#include "runtime/NameError.hpp"

using namespace py;

PyResult LoadName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	if (auto result = interpreter.get_object(m_object_name); result.is_ok()) {
		vm.reg(m_destination) = result.unwrap();
		return result;
	} else {
		return result;
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
