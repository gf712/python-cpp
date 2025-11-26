#include "LoadGlobal.hpp"

#include "interpreter/Interpreter.hpp"
#include "runtime/KeyError.hpp"
#include "runtime/NameError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyType.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> LoadGlobal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto *globals = interpreter.execution_frame()->globals();
	const auto &builtins = interpreter.execution_frame()->builtins()->symbol_table()->map();
	const auto &object_name = interpreter.execution_frame()->names(m_object_name);


	PyString *name = nullptr;

	if (auto *g = as<PyDict>(globals)) {
		if (const auto &it = g->map().find(String{ object_name }); it != g->map().end()) {
			vm.reg(m_destination) = it->second;
			return Ok(it->second);
		}
	} else {
		auto name_ = PyString::create(object_name);
		if (name_.is_err()) { return Err(name_.unwrap_err()); }
		name = name_.unwrap();
		auto it = [&] {
			[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
			return globals->as_mapping().unwrap().getitem(name);
		}();
		if (it.is_ok()) {
			vm.reg(m_destination) = it.unwrap();
			return Ok(it.unwrap());
		} else if (!it.unwrap_err()->type()->issubclass(KeyError::class_type())) {
			return Err(it.unwrap_err());
		}
	}

	if (!name) {
		auto name_ = PyString::create(object_name);
		if (name_.is_err()) { return Err(name_.unwrap_err()); }
		name = name_.unwrap();
	}

	if (const auto &it = builtins.find(name); it != builtins.end()) {
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
