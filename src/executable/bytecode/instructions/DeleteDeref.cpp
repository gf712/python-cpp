module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> DeleteDeref::execute(VirtualMachine &, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->freevars().size() > m_src);
	interpreter.execution_frame()->freevars()[m_src]->set_cell(nullptr);
	return Ok(py_none());
}

std::vector<uint8_t> DeleteDeref::serialize() const
{
	return {
		DELETE_DEREF,
		m_src,
	};
}
