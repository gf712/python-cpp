#include "LoadGlobal.hpp"

#include "runtime/NameError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"

using namespace py;

PyResult LoadGlobal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &globals = interpreter.execution_frame()->globals()->map();
	const auto &builtins = interpreter.execution_frame()->builtins()->symbol_table();

	if (const auto &it = globals.find(String{ m_object_name }); it != globals.end()) {
		vm.reg(m_destination) = it->second;
		return PyResult::Ok(it->second);
	}

	auto name = PyString::create(m_object_name);
	if (name.is_err()) { return name; }

	if (const auto &it = builtins.find(name.unwrap_as<PyString>()); it != builtins.end()) {
		vm.reg(m_destination) = it->second;
		return PyResult::Ok(it->second);
	}

	return PyResult::Err(name_error("name '{:s}' is not defined", m_object_name));
}

std::vector<uint8_t> LoadGlobal::serialize() const
{
	TODO();
	return {
		LOAD_GLOBAL,
		m_destination,
	};
}