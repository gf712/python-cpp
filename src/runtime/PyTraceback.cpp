#include "PyTraceback.hpp"
#include "MemoryError.hpp"
#include "runtime/PyFrame.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

PyTraceback::PyTraceback(PyType *type) : PyBaseObject(type) {}

std::string PyTraceback::to_string() const
{
	return fmt::format("<traceback object at {}>", static_cast<const void *>(this));
}

void PyTraceback::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_tb_frame) visitor.visit(*m_tb_frame);
	if (m_tb_next) visitor.visit(*m_tb_next);
}

PyResult<PyObject *> PyTraceback::__repr__() const { return PyString::create(to_string()); }

PyTraceback::PyTraceback(PyFrame *tb_frame, size_t tb_lasti, size_t tb_lineno, PyTraceback *tb_next)
	: PyBaseObject(types::BuiltinTypes::the().traceback()), m_tb_frame(tb_frame), m_tb_lasti(tb_lasti),
	  m_tb_lineno(tb_lineno), m_tb_next(tb_next)
{}

PyResult<PyTraceback *>
	PyTraceback::create(PyFrame *tb_frame, size_t tb_lasti, size_t tb_lineno, PyTraceback *tb_next)
{
	auto *obj =
		VirtualMachine::the().heap().allocate<PyTraceback>(tb_frame, tb_lasti, tb_lineno, tb_next);
	if (!obj) return Err(memory_error(sizeof(PyTraceback)));
	return Ok(obj);
}

PyType *PyTraceback::static_type() const { return types::traceback(); }

namespace {

	std::once_flag traceback_flag;

	std::unique_ptr<TypePrototype> register_traceback()
	{
		return std::move(klass<PyTraceback>("traceback")
							 .attr("tb_frame", &PyTraceback::m_tb_frame)
							 .attr("tb_next", &PyTraceback::m_tb_next)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyTraceback::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(traceback_flag, []() { type = register_traceback(); });
		return std::move(type);
	};
}

template<> PyTraceback *as(PyObject *obj)
{
	if (obj->type() == types::traceback()) { return static_cast<PyTraceback *>(obj); }
	return nullptr;
}

template<> const PyTraceback *as(const PyObject *obj)
{
	if (obj->type() == types::traceback()) { return static_cast<const PyTraceback *>(obj); }
	return nullptr;
}
}// namespace py
