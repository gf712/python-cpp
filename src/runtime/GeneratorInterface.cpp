#include "GeneratorInterface.hpp"
#include "PyAsyncGenerator.hpp"
#include "PyCode.hpp"
#include "PyCoroutine.hpp"
#include "PyFrame.hpp"
#include "PyGenerator.hpp"
#include "PyNone.hpp"
#include "PyTraceback.hpp"
#include "PyType.hpp"
#include "RuntimeError.hpp"
#include "StopIteration.hpp"
#include "ValueError.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

namespace py {

template<typename T>
GeneratorInterface<T>::GeneratorInterface(TypePrototype &type,
	PyFrame *frame,
	std::unique_ptr<StackFrame> &&stack_frame,
	bool is_running,
	PyObject *code,
	PyString *name,
	PyString *qualname)
	: PyBaseObject(type), m_frame(frame), m_stack_frame(std::move(stack_frame)),
	  m_is_running(is_running), m_code(code), m_name(name), m_qualname(qualname)
{}

template<typename T> std::string GeneratorInterface<T>::to_string() const
{
	// FIXME: use qualname
	return fmt::format("<{} object {} at {}>",
		T::GeneratorTypeName,
		m_name->value(),
		static_cast<const void *>(this));
}

template<typename T> PyResult<PyObject *> GeneratorInterface<T>::__repr__() const
{
	return PyString::create(to_string());
}

template<typename T> PyResult<PyObject *> GeneratorInterface<T>::__iter__() const
{
	return Ok(static_cast<T *>(const_cast<GeneratorInterface<T> *>(this)));
}

template<typename T> PyResult<PyObject *> GeneratorInterface<T>::__next__() { return send(); }

template<typename T> PyResult<PyObject *> GeneratorInterface<T>::send()
{
	if (m_is_running) { return Err(value_error("generator already executing")); }
	if (m_frame == nullptr) {
		// exhausted generator, or raised an exception
		return Err(stop_iteration(py_none()));
	}

	m_is_running = true;

	ASSERT(m_code);
	ASSERT(as<PyCode>(m_code));

	m_frame->m_f_back = VirtualMachine::the().interpreter().execution_frame();
	auto result = VirtualMachine::the().interpreter().call(
		as<PyCode>(m_code)->function(), m_frame, *m_stack_frame);

	m_is_running = false;
	m_frame->m_f_back = nullptr;

	if (m_invalid_return && result.unwrap_err()->type()->issubclass(stop_iteration()->type())) {
		spdlog::debug("generator returned value {}", result.unwrap_err()->args()->to_string());
	} else if (result.is_ok()) {
		spdlog::debug("generator result {}", result.unwrap()->to_string());
	} else if (result.is_err()
			   && result.unwrap_err()->type()->issubclass(stop_iteration()->type())) {
		result = Err(runtime_error("generator raise StopIteration"));
	}

	if (result.is_err()) { m_frame = nullptr; }

	m_invalid_return = false;
	return result;
}

template<typename T> PyResult<PyObject *> GeneratorInterface<T>::close()
{
	// TODO: implement generator close logic
	return Ok(py_none());
}

template<typename T> void GeneratorInterface<T>::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_frame) visitor.visit(*m_frame);
	if (m_code) visitor.visit(*m_code);
	if (m_name) visitor.visit(*m_name);
	if (m_qualname) visitor.visit(*m_qualname);
	if (m_exception_stack) {
		for (const auto &exc : *m_exception_stack) {
			if (exc.exception) visitor.visit(*exc.exception);
			if (exc.exception_type) visitor.visit(*exc.exception_type);
			if (exc.traceback) visitor.visit(*exc.traceback);
		}
	}

	if (m_stack_frame) {
		for (const auto &el : m_stack_frame->registers) {
			if (std::holds_alternative<PyObject *>(el)) {
				auto *obj = std::get<PyObject *>(el);
				if (obj) { visitor.visit(*obj); }
			}
		}

		for (const auto &el : m_stack_frame->locals) {
			if (std::holds_alternative<PyObject *>(el)) {
				auto *obj = std::get<PyObject *>(el);
				if (obj) { visitor.visit(*obj); }
			}
		}
	}
}

template class GeneratorInterface<PyAsyncGenerator>;
template class GeneratorInterface<PyCoroutine>;
template class GeneratorInterface<PyGenerator>;

}// namespace py