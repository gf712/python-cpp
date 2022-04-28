#include "LoadBuildClass.hpp"

#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"

using namespace py;

PyResult LoadBuildClass::execute(VirtualMachine &vm, Interpreter &intepreter) const
{
	auto str = PyString::create("__build_class__");
	if (str.is_err()) { return str; }
	auto result = std::get<PyObject *>(
		intepreter.execution_frame()->builtins()->symbol_table().at(str.unwrap_as<PyString>()));
	vm.reg(m_dst) = result;
	return PyResult::Ok(result);
}


std::vector<uint8_t> LoadBuildClass::serialize() const
{
	return {
		LOAD_ATTR,
		m_dst,
	};
}