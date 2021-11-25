#include "ImportName.hpp"

#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"


void ImportName::execute(VirtualMachine &vm, Interpreter &) const
{
	std::string name = std::accumulate(
		std::next(m_names.begin()), m_names.end(), *m_names.begin(), [](auto rhs, auto lhs) {
			return std::move(rhs) + "." + lhs;
		});

	auto module_name = vm.heap().allocate<PyString>(name);
	if (auto *module = PyModule::create(module_name)) { vm.reg(m_destination) = module; }
}