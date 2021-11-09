#include "LoadName.hpp"

#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"


void LoadName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	Value obj;

	const auto &name_value = String{ m_object_name };

	ASSERT(interpreter.execution_frame()->locals())
	ASSERT(interpreter.execution_frame()->globals())
	ASSERT(interpreter.execution_frame()->builtins())

	const auto &locals = interpreter.execution_frame()->locals()->map();
	const auto &globals = interpreter.execution_frame()->globals()->map();
	const auto &builtins = interpreter.execution_frame()->builtins()->symbol_table();

	auto pystr_name = PyString::create(m_object_name);

	if (auto it = locals.find(name_value); it != locals.end()) {
		obj = it->second;
	} else if (auto it = globals.find(name_value); it != globals.end()) {
		obj = it->second;
	} else if (auto it = builtins.find(pystr_name); it != builtins.end()) {
		obj = it->second;
	} else {
		interpreter.raise_exception("NameError: name '{:s}' is not defined", m_object_name);
		return;
	}
	vm.reg(m_destination) = obj;
}