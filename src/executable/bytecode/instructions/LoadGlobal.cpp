#include "LoadGlobal.hpp"

#include "interpreter/Interpreter.hpp"
#include "runtime/NameError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyModule.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LoadGlobal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &globals = interpreter.execution_frame()->globals()->map();
	const auto &builtins = interpreter.execution_frame()->builtins()->symbol_table()->map();
	const auto &object_name = interpreter.execution_frame()->names(m_object_name);

	if (const auto &it = globals.find(String{ object_name }); it != globals.end()) {
		vm.reg(m_destination) = it->second;
		return Ok(it->second);
	}

	auto name = PyString::create(object_name);
	if (name.is_err()) { return Err(name.unwrap_err()); }

	if (const auto &it = builtins.find(name.unwrap()); it != builtins.end()) {
		vm.reg(m_destination) = it->second;
		return Ok(it->second);
	}

	return Err(name_error("name '{:s}' is not defined", object_name));
}

std::vector<uint8_t> LoadGlobal::serialize() const
{
	return {
		LOAD_GLOBAL,
		m_destination,
		m_object_name,
	};
}