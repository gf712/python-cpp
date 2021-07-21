#include "Interpreter.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"

#include <iostream>

Interpreter::Interpreter() : m_current_frame(ExecutionFrame::create(nullptr)) {}

void Interpreter::setup()
{
	allocate_object<PyNativeFunction>("print", [this](const std::shared_ptr<PyObject> &arg) {
		std::cout << arg->repr_impl(*this)->to_string() << '\n';
		return py_none();
	});
}