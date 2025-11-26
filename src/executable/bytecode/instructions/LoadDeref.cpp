#include "LoadDeref.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/NameError.hpp"
#include "runtime/PyCell.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyObject.hpp"
#include "vm/VM.hpp"
#include <variant>

using namespace py;

PyResult<Value> LoadDeref::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.execution_frame()->freevars().size() > m_source);
	ASSERT(interpreter.execution_frame()->freevars()[m_source]);
	auto result = interpreter.execution_frame()->freevars()[m_source]->content();
	if (std::holds_alternative<PyObject *>(result) && !std::get<PyObject *>(result)) {
		auto *code = interpreter.execution_frame()->code();
		ASSERT(m_source < code->m_freevars.size());
		const auto &name = code->m_freevars[m_source];
		return Err(
			name_error("free variable '{}' referenced before assignment in enclosing scope", name));
	}
	vm.reg(m_destination) = result;
	return Ok(result);
}

std::vector<uint8_t> LoadDeref::serialize() const
{
	return {
		LOAD_DEREF,
		m_destination,
		m_source,
	};
}