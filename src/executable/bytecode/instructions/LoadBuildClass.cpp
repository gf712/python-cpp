#include "LoadBuildClass.hpp"

#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"

using namespace py;

PyResult<Value> LoadBuildClass::execute(VirtualMachine &vm, Interpreter &intepreter) const
{
	auto str = PyString::create("__build_class__");
	if (str.is_err()) { return Err(str.unwrap_err()); }
	auto result =
		PyObject::from(intepreter.execution_frame()->builtins()->symbol_table().at(str.unwrap()));
	if (result.is_err()) return Err(result.unwrap_err());
	vm.reg(m_dst) = result.unwrap();
	return Ok(Value{ result.unwrap() });
}


std::vector<uint8_t> LoadBuildClass::serialize() const
{
	return {
		LOAD_ATTR,
		m_dst,
	};
}