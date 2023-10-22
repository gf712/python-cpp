#include "DeleteGlobal.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyString.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> DeleteGlobal::execute(VirtualMachine &, Interpreter &interpreter) const
{
	auto name = interpreter.execution_frame()->consts(m_name);
	auto name_str = PyObject::from(name);
	if (name_str.is_err()) return name_str;
	return interpreter.execution_frame()
		->globals()
		->delete_item(name_str.unwrap())
		.and_then([](auto) { return Ok(py_none()); });
}

std::vector<uint8_t> DeleteGlobal::serialize() const
{
	return {
		DELETE_GLOBAL,
		m_name,
	};
}
