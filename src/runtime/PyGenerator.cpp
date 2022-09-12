#include "PyGenerator.hpp"
#include "MemoryError.hpp"
#include "PyCode.hpp"
#include "PyFrame.hpp"
#include "PyTraceback.hpp"
#include "RuntimeError.hpp"
#include "StopIteration.hpp"
#include "ValueError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyGenerator *as(PyObject *obj)
{
	if (obj->type() == generator()) { return static_cast<PyGenerator *>(obj); }
	return nullptr;
}

template<> const PyGenerator *as(const PyObject *obj)
{
	if (obj->type() == generator()) { return static_cast<const PyGenerator *>(obj); }
	return nullptr;
}

PyGenerator::PyGenerator(PyFrame *frame,
	std::unique_ptr<StackFrame> &&stack_frame,
	bool is_running,
	PyObject *code,
	PyString *name,
	PyString *qualname)
	: PyBaseObject(BuiltinTypes::the().generator()), m_frame(frame),
	  m_stack_frame(std::move(stack_frame)), m_is_running(is_running), m_code(code), m_name(name),
	  m_qualname(qualname)
{}

PyResult<PyGenerator *> PyGenerator::create(PyFrame *frame,
	std::unique_ptr<StackFrame> &&stack_frame,
	PyString *name,
	PyString *qualname)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyGenerator>(
			frame, std::move(stack_frame), false, frame->code(), name, qualname)) {
		return Ok(obj);
	}
	return Err(memory_error(sizeof(PyGenerator)));
}

std::string PyGenerator::to_string() const
{
	// FIXME: use qualname
	return fmt::format(
		"<generator object {} at {}>", m_name->value(), static_cast<const void *>(this));
}

PyResult<PyObject *> PyGenerator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyGenerator::__iter__() const { return Ok(const_cast<PyGenerator *>(this)); }

PyResult<PyObject *> PyGenerator::__next__() { return send(); }

PyResult<PyObject *> PyGenerator::send()
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

namespace {
	std::once_flag generator_flag;

	std::unique_ptr<TypePrototype> register_generator()
	{
		return std::move(klass<PyGenerator>("generator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyGenerator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(generator_flag, []() { type = register_generator(); });
		return std::move(type);
	};
}

PyType *PyGenerator::type() const { return generator(); }

void PyGenerator::visit_graph(Visitor &visitor)
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
				if (obj) obj->visit_graph(visitor);
			}
		}

		for (const auto &el : m_stack_frame->locals) {
			if (std::holds_alternative<PyObject *>(el)) {
				auto *obj = std::get<PyObject *>(el);
				if (obj) obj->visit_graph(visitor);
			}
		}
	}
}

}// namespace py