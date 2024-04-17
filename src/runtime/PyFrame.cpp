#include "PyFrame.hpp"
#include "BaseException.hpp"
#include "KeyError.hpp"
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
	if (obj->type() == types::frame()) { return static_cast<PyFrame *>(obj); }
	return nullptr;
}

template<> const PyFrame *as(const PyObject *obj)
{
	if (obj->type() == types::frame()) { return static_cast<const PyFrame *>(obj); }
	return nullptr;
}

PyFrame::PyFrame(PyType *type) : PyBaseObject(type) {}

PyFrame::PyFrame(const std::vector<std::string> names)
	: PyBaseObject(types::BuiltinTypes::the().frame()), m_names(std::move(names))
{}

PyFrame *PyFrame::create(PyFrame *parent,
	size_t register_count,
	size_t free_vars_count,
	PyCode *code,
	PyObject *globals,
	PyObject *locals,
	const PyTuple *consts,
	const std::vector<std::string> names,
	PyObject *generator)
{
	// TODO: handle wrong type here or somewhere else
	if (globals->as_mapping().is_err()) { TODO(); }
	if (locals->as_mapping().is_err()) { TODO(); }
	if (!globals->type()->underlying_type().mapping_type_protocol->__getitem__.has_value()) {
		TODO();
	}
	if (!locals->type()->underlying_type().mapping_type_protocol->__getitem__.has_value()) {
		TODO();
	}

	auto *new_frame = VirtualMachine::the().heap().allocate<PyFrame>(std::move(names));
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
		auto builtins = [new_frame]() -> PyResult<PyObject *> {
			if (auto *locals = as<PyDict>(new_frame->m_locals)) {
				if (auto value = (*locals)[String{ "__builtins__" }]; value.has_value()) {
					return PyObject::from(*value);
				} else {
					return Err(key_error("__builtins__"));
				}
			} else {
				return new_frame->m_locals->as_mapping().and_then([](PyMappingWrapper mapping) {
					return mapping.getitem(PyString::create("__builtins__").unwrap());
				});
			}
		}();
		ASSERT(builtins.is_ok());
		ASSERT(as<PyModule>(builtins.unwrap()));
		// TODO: could this just return the builtin singleton?
		new_frame->m_builtins = as<PyModule>(builtins.unwrap());
		new_frame->m_exception_stack = std::make_shared<std::vector<ExceptionStackItem>>();
	}

	new_frame->m_generator = generator;
	return new_frame;
}

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
	spdlog::debug("PyFrame::pop_exception: @{}", static_cast<void*>(this));
	spdlog::debug("PyFrame::pop_exception: current exception count {}", m_exception_stack->size());
	spdlog::debug("PyFrame::pop_exception: Popped exception {}", exception->to_string());
	m_exception_stack->pop_back();
	spdlog::debug(
		"PyFrame::pop_exception: cleared exception, {} exceptions left", m_exception_stack->size());
	return exception;
}

PyResult<std::monostate> PyFrame::put_local(const std::string &name, const Value &value)
{
	if (auto *locals = as<PyDict>(m_locals)) {
		locals->insert(String{ name }, value);
		return Ok(std::monostate{});
	} else {
		return m_locals->as_mapping().unwrap().setitem(
			PyString::create(name).unwrap(), PyObject::from(value).unwrap());
	}
}

PyResult<std::monostate> PyFrame::put_global(const std::string &name, const Value &value)
{
	if (auto *globals = as<PyDict>(m_globals)) {
		globals->insert(String{ name }, value);
		return Ok(std::monostate{});
	} else {
		return m_globals->as_mapping().unwrap().setitem(
			PyString::create(name).unwrap(), PyObject::from(value).unwrap());
	}
}

PyObject *PyFrame::locals() const
{
	auto insert = [this](const Value &key, const Value &value) {
		if (auto l = as<PyDict>(m_locals)) {
			l->insert(key, value);
		} else {
			m_locals->setitem(PyObject::from(key).unwrap(), PyObject::from(value).unwrap());
		}
	};
	auto remove = [this](const Value &key) {
		if (auto l = as<PyDict>(m_locals)) {
			l->remove(key);
		} else {
			m_locals->delitem(PyObject::from(key).unwrap());
		}
	};
	const auto &fast_locals = VirtualMachine::the().stack_locals();
	for (size_t i = 0; const auto &varname : m_f_code->m_varnames) {
		ASSERT(i < fast_locals.size());
		const auto &value = fast_locals[i++];
		if (std::holds_alternative<PyObject *>(value) && !std::get<PyObject *>(value)) {
			remove(String{ varname });
		} else {
			insert(String{ varname }, value);
		}
	}

	size_t i = 0;
	for (const auto &cell_name : m_f_code->m_cellvars) {
		const auto &cell = freevars()[i++];
		if (!cell || cell->empty()) {
			remove(String{ cell_name });
		} else {
			insert(String{ cell_name }, cell->content());
		}
	}

	if (!m_f_code->flags().is_set(CodeFlags::Flag::CLASS)) {
		for (const auto &freevar_name : m_f_code->m_freevars) {
			const auto &cell = freevars()[i++];
			if (!cell || cell->empty()) {
				remove(String{ freevar_name });
			} else {
				insert(String{ freevar_name }, cell->content());
			}
		}
	}

	return m_locals;
}

PyObject *PyFrame::globals() const { return m_globals; }

PyModule *PyFrame::builtins() const { return m_builtins; }

const std::vector<PyCell *> &PyFrame::freevars() const { return m_freevars; }
std::vector<PyCell *> &PyFrame::freevars() { return m_freevars; }

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

PyType *PyFrame::static_type() const { return types::frame(); }

}// namespace py
