#include "PyFrame.hpp"
#include "PyCell.hpp"
#include "PyDict.hpp"
#include "PyModule.hpp"
#include "PyNone.hpp"
#include "PyObject.hpp"
#include "PyTraceback.hpp"
#include "PyType.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

PyFrame::PyFrame() : PyBaseObject(BuiltinTypes::the().frame()) {}

PyFrame *PyFrame::create(PyFrame *parent,
	size_t register_count,
	size_t free_vars_count,
	PyDict *globals,
	PyDict *locals,
	const PyTuple *consts)
{
	auto *new_frame = Heap::the().allocate<PyFrame>();
	new_frame->m_f_back = parent;
	new_frame->m_register_count = register_count;
	new_frame->m_globals = globals;
	new_frame->m_locals = locals;
	new_frame->m_consts = consts;
	// if (parent) {
	// 	new_frame->m_freevars = parent->m_freevars;
	// 	new_frame->m_freevars.resize(parent->m_freevars.size() + free_vars_count, nullptr);
	// } else {
	// 	new_frame->m_freevars = std::vector<PyCell *>(free_vars_count, nullptr);
	// }
	new_frame->m_freevars = std::vector<PyCell *>(free_vars_count, nullptr);

	if (new_frame->m_f_back) {
		new_frame->m_builtins = new_frame->m_f_back->m_builtins;
	} else {
		ASSERT(new_frame->locals()->map().contains(String{ "__builtins__" }))
		ASSERT(std::get<PyObject *>((*new_frame->m_locals)[String{ "__builtins__" }])->type()
			   == module())
		// TODO: could this just return the builtin singleton?
		new_frame->m_builtins =
			as<PyModule>(std::get<PyObject *>((*new_frame->m_locals)[String{ "__builtins__" }]));
	}
	return new_frame;
}

void PyFrame::set_exception_to_catch(PyObject *exception) { m_exception_to_catch = exception; }

void PyFrame::set_exception(PyObject *exception)
{
	if (exception) {
		// make sure that we are not accidentally overriding an active exception with another
		// exception
		ASSERT(m_exception_stack.empty())
		m_exception_stack.push_back(ExceptionStackItem{
			.exception = exception, .exception_type = exception->type(), .traceback = nullptr });
	} else {
		// FIXME: create a PyFrame::clear_exception method instead
		m_exception_stack.pop_back();
	}
}

void PyFrame::clear_stashed_exception() { m_stashed_exception.reset(); }

void PyFrame::stash_exception()
{
	m_stashed_exception = m_exception_stack.back();
	m_exception_stack.pop_back();
}

bool PyFrame::catch_exception(PyObject *exception) const
{
	if (m_exception_to_catch)
		return exception->type()->issubclass(m_exception_to_catch->type());
	else
		return false;
}

void PyFrame::put_local(const std::string &name, const Value &value)
{
	m_locals->insert(String{ name }, value);
}

void PyFrame::put_global(const std::string &name, const Value &value)
{
	m_globals->insert(String{ name }, value);
}

PyDict *PyFrame::locals() const { return m_locals; }
PyDict *PyFrame::globals() const { return m_globals; }
PyModule *PyFrame::builtins() const { return m_builtins; }

const std::vector<py::PyCell *> &PyFrame::freevars() const { return m_freevars; }
std::vector<py::PyCell *> &PyFrame::freevars() { return m_freevars; }

PyFrame *PyFrame::exit() { return m_f_back; }

std::string PyFrame::to_string() const
{
	const auto locals = m_locals ? m_locals->to_string() : "";
	const auto globals = m_globals ? m_globals->to_string() : "";
	const auto builtins = m_builtins ? m_builtins->to_string() : "";
	const void *parent = m_f_back ? &m_f_back : nullptr;

	return fmt::format("PyFrame(locals={}, globals={}, builtins={}, parent={})",
		locals,
		globals,
		builtins,
		parent);
}

void PyFrame::visit_graph(Visitor &visitor)
{
	visitor.visit(*this);
	if (m_locals) visitor.visit(*m_locals);
	if (m_globals) visitor.visit(*m_globals);
	if (m_builtins) visitor.visit(*m_builtins);
	if (m_exception_to_catch) visitor.visit(*m_exception_to_catch);
	for (const auto &exception_stack_item : m_exception_stack) {
		if (exception_stack_item.exception) { visitor.visit(*exception_stack_item.exception); }
		if (exception_stack_item.exception_type) {
			visitor.visit(*exception_stack_item.exception_type);
		}
		if (exception_stack_item.traceback) { visitor.visit(*exception_stack_item.traceback); }
	}
	if (m_stashed_exception.has_value()) visitor.visit(*m_stashed_exception->exception);
	if (m_f_back) { visitor.visit(*m_f_back); }
	for (const auto &freevar : m_freevars) {
		if (freevar) { visitor.visit(*freevar); }
	}
	if (m_consts) { visitor.visit(*const_cast<PyTuple *>(m_consts)); }
}

Value PyFrame::consts(size_t index) const
{
	ASSERT(index < m_consts->size())
	spdlog::debug("m_consts: {}", (void *)m_consts);
	return m_consts->elements()[index];
}

namespace {

	std::once_flag frame_flag;

	std::unique_ptr<TypePrototype> register_frame()
	{
		return std::move(klass<PyFrame>("frame").type);
	}
}// namespace

std::unique_ptr<TypePrototype> PyFrame::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(frame_flag, []() { type = register_frame(); });
	return std::move(type);
}


}// namespace py
