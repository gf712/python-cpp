#include "PyFrame.hpp"
#include "BaseException.hpp"
#include "PyCell.hpp"
#include "PyCode.hpp"
#include "PyDict.hpp"
#include "PyModule.hpp"
#include "PyNone.hpp"
#include "PyObject.hpp"
#include "PyTraceback.hpp"
#include "PyType.hpp"
#include "executable/Function.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> PyFrame *as(PyObject *obj)
{
	if (obj->type() == frame()) { return static_cast<PyFrame *>(obj); }
	return nullptr;
}

template<> const PyFrame *as(const PyObject *obj)
{
	if (obj->type() == frame()) { return static_cast<const PyFrame *>(obj); }
	return nullptr;
}

PyFrame::PyFrame(const std::vector<std::string> &names)
	: PyBaseObject(BuiltinTypes::the().frame()), m_names(names)
{}

PyFrame *PyFrame::create(PyFrame *parent,
	size_t register_count,
	size_t free_vars_count,
	PyCode *code,
	PyDict *globals,
	PyDict *locals,
	const PyTuple *consts,
	const std::vector<std::string> &names,
	PyObject *generator)
{
	auto *new_frame = Heap::the().allocate<PyFrame>(names);
	new_frame->m_f_back = parent;
	new_frame->m_register_count = register_count;
	new_frame->m_globals = globals;
	new_frame->m_locals = locals;
	new_frame->m_consts = consts;
	new_frame->m_f_code = code;
	new_frame->m_freevars = std::vector<PyCell *>(free_vars_count, nullptr);

	if (new_frame->m_f_back) {
		new_frame->m_builtins = new_frame->m_f_back->m_builtins;
		new_frame->m_exception_stack = new_frame->m_f_back->m_exception_stack;
	} else {
		ASSERT(new_frame->locals()->map().contains(String{ "__builtins__" }))
		ASSERT(std::get<PyObject *>((*new_frame->m_locals)[String{ "__builtins__" }])->type()
			   == py::module())
		// TODO: could this just return the builtin singleton?
		new_frame->m_builtins =
			as<PyModule>(std::get<PyObject *>((*new_frame->m_locals)[String{ "__builtins__" }]));
		new_frame->m_exception_stack = std::make_shared<std::vector<ExceptionStackItem>>();
	}

	new_frame->m_generator = generator;
	return new_frame;
}

void PyFrame::set_exception_to_catch(BaseException *exception) { m_exception_to_catch = exception; }

void PyFrame::push_exception(BaseException *exception)
{
	ASSERT(exception)
	spdlog::debug("PyFrame::push_exception: current exception count {}", m_exception_stack->size());
	m_exception_stack->push_back(ExceptionStackItem{ .exception = exception,
		.exception_type = exception->type(),
		.traceback = exception->traceback() });
	spdlog::debug("PyFrame::push_exception: pushed exception {}",
		m_exception_stack->back().exception->to_string());
	spdlog::debug("PyFrame::push_exception: added exception, stack has now {} exceptions",
		m_exception_stack->size());
}

BaseException *PyFrame::pop_exception()
{
	ASSERT(!m_exception_stack->empty())
	auto *exception = m_exception_stack->back().exception;
	spdlog::debug("PyFrame::pop_exception: current exception count {}", m_exception_stack->size());
	spdlog::debug("PyFrame::pop_exception: Popped exception {}", exception->to_string());
	m_exception_stack->pop_back();
	spdlog::debug(
		"PyFrame::pop_exception: cleared exception, {} exceptions left", m_exception_stack->size());
	return exception;
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

PyFrame *PyFrame::exit()
{
	spdlog::debug("Leaving PyFrame '{}' and entering PyFrame '{}'",
		m_f_code->function()->function_name(),
		m_f_back->m_f_code->function()->function_name());

	return m_f_back;
}

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
	PyObject::visit_graph(visitor);
	if (m_locals) visitor.visit(*m_locals);
	if (m_globals) visitor.visit(*m_globals);
	if (m_builtins) visitor.visit(*m_builtins);
	if (m_f_code) visitor.visit(*m_f_code);
	if (m_exception_to_catch) visitor.visit(*m_exception_to_catch);
	for (const auto &exception_stack_item : *m_exception_stack) {
		if (exception_stack_item.exception) { visitor.visit(*exception_stack_item.exception); }
		if (exception_stack_item.exception_type) {
			visitor.visit(*exception_stack_item.exception_type);
		}
		if (exception_stack_item.traceback) { visitor.visit(*exception_stack_item.traceback); }
	}
	if (m_f_back) { visitor.visit(*m_f_back); }
	for (const auto &freevar : m_freevars) {
		if (freevar) { visitor.visit(*freevar); }
	}
	if (m_consts) { visitor.visit(*const_cast<PyTuple *>(m_consts)); }
	if (m_generator) { visitor.visit(*m_generator); }
}

Value PyFrame::consts(size_t index) const
{
	ASSERT(index < m_consts->size())
	spdlog::debug("m_consts: {}", (void *)m_consts);
	return m_consts->elements()[index];
}

const std::string &PyFrame::names(size_t index) const
{
	ASSERT(index < m_names.size())
	return m_names[index];
}

namespace {

	std::once_flag frame_flag;

	std::unique_ptr<TypePrototype> register_frame()
	{
		return std::move(klass<PyFrame>("frame").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyFrame::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(frame_flag, []() { type = register_frame(); });
		return std::move(type);
	};
}

PyType *PyFrame::type() const { return frame(); }

}// namespace py
