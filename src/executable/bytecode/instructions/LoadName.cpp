#include "LoadName.hpp"

#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"


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

	result.reserve(result.size() + sizeof(size_t) + m_object_name.size());

	const size_t &name_size = m_object_name.size();
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		result.push_back(reinterpret_cast<const uint8_t *>(&name_size)[i]);
	}

	for (const auto &c : m_object_name) {
		result.push_back(*reinterpret_cast<const uint8_t *>(&c));
	}

	return result;
}
