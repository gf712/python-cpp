#include "ImportName.hpp"

#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"

using namespace py;

PyResult ImportName::execute(VirtualMachine &vm, Interpreter &) const
{
	std::string name = std::accumulate(
		std::next(m_names.begin()), m_names.end(), *m_names.begin(), [](auto rhs, auto lhs) {
			return std::move(rhs) + "." + lhs;
		});

	auto module_name = PyString::create(name);
	if (module_name.is_err()) { return module_name; }
	auto module = PyModule::create(module_name.unwrap_as<PyString>());
	if (module.is_ok()) { vm.reg(m_destination) = module.unwrap(); }
	return module;
}

std::vector<uint8_t> ImportName::serialize() const
{
	TODO();
	return {
		IMPORT_NAME,
		m_destination,
	};
}