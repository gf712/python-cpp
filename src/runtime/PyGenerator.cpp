#include "PyGenerator.hpp"
#include "MemoryError.hpp"
#include "PyCode.hpp"
#include "PyFrame.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

PyGenerator::PyGenerator(PyFrame *frame,
	bool is_running,
	PyObject *code,
	PyString *name,
	PyString *qualname)
	: PyBaseObject(BuiltinTypes::the().generator()), m_frame(frame), m_is_running(is_running),
	  m_code(code), m_name(name), m_qualname(qualname)
{}

PyResult<PyGenerator *> PyGenerator::create(PyFrame *frame, PyString *name, PyString *qualname)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyGenerator>(frame, false, frame->code(), name, qualname)) {
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

PyResult<PyObject *> PyGenerator::__next__()
{
	TODO();
	return Ok(nullptr);
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
	if (m_frame) visitor.visit(*m_frame);
	if (m_code) visitor.visit(*m_code);
	if (m_name) visitor.visit(*m_name);
	if (m_qualname) visitor.visit(*m_qualname);
}

}// namespace py