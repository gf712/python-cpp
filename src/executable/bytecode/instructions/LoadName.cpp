#include "LoadName.hpp"
#include "executable/bytecode/serialization/serialize.hpp"


void LoadName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	if (auto value = interpreter.get_object(m_object_name)) { vm.reg(m_destination) = *value; }
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
