#include "Interpreter.hpp"

#include "runtime/BaseException.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyBuiltins.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"

#include <iostream>

Interpreter::Interpreter() {}

void Interpreter::setup()
{
	auto globals = VirtualMachine::the().heap().allocate<PyDict>(
		PyDict::MapType{ { String{ "__builtins__" }, fetch_builtins(*this) },
			{ String{ "__name__" }, PyString::create("__main__") },
			{ String{ "__doc__" }, py_none() },
			{ String{ "__package__" }, py_none() } });
	auto locals = std::make_unique<PyDict>(globals->map());
	m_current_frame = ExecutionFrame::create(nullptr, globals, std::move(locals), nullptr);
	m_global_frame = m_current_frame;
}

void Interpreter::unwind()
{
	auto raised_exception = m_current_frame->exception();
	while (!m_current_frame->catch_exception(raised_exception)) {
		// don't unwind beyond the main frame
		if (!m_current_frame->parent()) {
			// uncaught exception
			std::cout
				<< std::static_pointer_cast<BaseException>(m_current_frame->exception())->what()
				<< '\n';
			break;
		}
		m_current_frame = m_current_frame->exit();
	}
	m_current_frame->set_exception(nullptr);
}