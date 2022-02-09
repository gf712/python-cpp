#include "LoadGlobal.hpp"

#include "runtime/NameError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"

using namespace py;

void LoadGlobal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &globals = interpreter.execution_frame()->globals()->map();
	const auto &builtins = interpreter.execution_frame()->builtins()->symbol_table();

	if (const auto &it = globals.find(String{ m_object_name }); it != globals.end()) {
		vm.reg(m_destination) = it->second;
		return;
	}

	auto *name = PyString::create(m_object_name);
	if (const auto &it = builtins.find(name); it != builtins.end()) {
		vm.reg(m_destination) = it->second;
		return;
	}

	interpreter.raise_exception(name_error("name '{:s}' is not defined", m_object_name));
}