#include "PyFunction.hpp"
#include "PyBoundMethod.hpp"
#include "PyCell.hpp"
#include "PyCode.hpp"
#include "PyDict.hpp"
#include "PyModule.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "TypeError.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

#include "utilities.hpp"

namespace py {

PyFunction::PyFunction(std::string name,
	std::vector<Value> defaults,
	std::vector<Value> kwonly_defaults,
	PyCode *code,
	std::vector<PyCell *> closure,
	PyDict *globals)
	: PyBaseObject(BuiltinTypes::the().function()), m_code(code), m_globals(globals),
	  m_defaults(std::move(defaults)), m_kwonly_defaults(std::move(kwonly_defaults)),
	  m_closure(std::move(closure))
{
	auto name_ = PyString::create(name);
	if (name_.is_err()) { TODO(); }
	m_name = name_.unwrap();

	auto dict_ = PyDict::create();
	if (dict_.is_err()) { TODO(); }
	m_dict = dict_.unwrap();
}

void PyFunction::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	m_code->visit_graph(visitor);
	if (m_globals) visitor.visit(*m_globals);
	if (m_module) m_module->visit_graph(visitor);
	if (m_dict) visitor.visit(*m_dict);
	if (m_name) visitor.visit(*m_name);
	for (const auto &c : m_closure) { c->visit_graph(visitor); }
}

PyType *PyFunction::type() const { return function(); }

PyResult<PyObject *> PyFunction::__repr__() const
{
	return PyString::create(fmt::format("<function {}>", m_name->value()));
}

PyResult<PyObject *> PyFunction::__get__(PyObject *instance, PyObject * /*owner*/) const
{
	if (!instance || instance == py_none()) { return Ok(const_cast<PyFunction *>(this)); }
	return PyBoundMethod::create(instance, const_cast<PyFunction *>(this));
}

PyResult<PyObject *>
	PyFunction::call_with_frame(PyDict *locals, PyTuple *args, PyDict *kwargs) const
{
	auto *function_frame = PyFrame::create(VirtualMachine::the().interpreter().execution_frame(),
		m_code->register_count(),
		m_code->cellvars_count() + m_code->freevars_count(),
		m_globals,
		locals,
		m_code->consts());
	[[maybe_unused]] auto scoped_stack =
		VirtualMachine::the().interpreter().setup_call_stack(m_code->function(), function_frame);

	for (size_t i = 0; i < m_code->cellvars_count(); ++i) {
		auto cell = PyCell::create();
		if (cell.is_err()) return cell;
		function_frame->freevars()[i] = cell.unwrap();
	}

	const auto &cell2arg = m_code->cell2arg();
	const auto &varnames = m_code->varnames();
	const size_t total_arguments_count = m_code->arg_count() + m_code->kwonly_arg_count();
	std::vector<std::string> positional_args{ varnames.begin(),
		varnames.begin() + m_code->arg_count() };
	std::vector<std::string> keyword_only_args{ varnames.begin() + m_code->arg_count(),
		varnames.begin() + total_arguments_count };

	size_t args_count = 0;
	size_t kwargs_count = 0;

	if (args) {
		size_t max_args = std::min(args->size(), m_code->arg_count());
		for (size_t idx = 0; idx < max_args; ++idx) {
			const auto &obj = args->elements()[idx];
			VirtualMachine::the().stack_local(idx) = obj;
			if (auto it = std::find(cell2arg.begin(), cell2arg.end(), idx); it != cell2arg.end()) {
				const auto free_var_idx = std::distance(cell2arg.begin(), it);
				auto cell = PyCell::create(obj);
				if (cell.is_err()) return cell;
				function_frame->freevars()[free_var_idx] = cell.unwrap();
			}
		}
		args_count = max_args;
	}
	if (kwargs) {
		const auto &argnames = m_code->varnames();
		for (const auto &[key, value] : kwargs->map()) {
			ASSERT(std::holds_alternative<String>(key))
			auto key_str = std::get<String>(key);
			auto arg_iter = std::find(argnames.begin(), argnames.end(), key_str.s);
			if (arg_iter == argnames.end()) {
				if (m_code->flags().is_set(CodeFlags::Flag::VARKEYWORDS)) {
					continue;
				} else {
					return Err(type_error("{}() got an unexpected keyword argument '{}'",
						m_name->value(),
						key_str.s));
				}
			}
			auto &arg =
				VirtualMachine::the().stack_local(std::distance(argnames.begin(), arg_iter));

			if (std::holds_alternative<PyObject *>(arg)) {
				if (std::get<PyObject *>(arg)) {
					return Err(type_error(
						"{}() got multiple values for argument '{}'", m_name->value(), key_str.s));
				}
			}
			if (auto it = std::find(cell2arg.begin(), cell2arg.end(), kwargs_count);
				it != cell2arg.end()) {
				const auto free_var_idx = std::distance(cell2arg.begin(), it);
				auto cell = PyCell::create(value);
				if (cell.is_err()) return cell;
				function_frame->freevars()[free_var_idx] = cell.unwrap();
			}
			arg = value;
			kwargs_count++;
		}
	}

	{
		const auto &defaults = m_defaults;
		auto default_iter = defaults.rbegin();
		for (size_t i = m_code->arg_count() - 1; i > (m_code->arg_count() - defaults.size() - 1);
			 --i) {
			auto &arg = VirtualMachine::the().stack_local(i);
			if (std::holds_alternative<PyObject *>(arg) && !std::get<PyObject *>(arg)) {
				VirtualMachine::the().stack_local(i) = *default_iter;
			}
			if (auto it = std::find(cell2arg.begin(), cell2arg.end(), i); it != cell2arg.end()) {
				const auto free_var_idx = std::distance(cell2arg.begin(), it);
				auto cell = PyCell::create(*default_iter);
				if (cell.is_err()) return cell;
				function_frame->freevars()[free_var_idx] = cell.unwrap();
			}
			default_iter = std::next(default_iter);
		}
	}
	{
		const auto &kw_defaults = m_kwonly_defaults;
		auto kw_default_iter = kw_defaults.rbegin();
		const size_t start = m_code->kwonly_arg_count() + m_code->arg_count() - 1;
		for (size_t i = start; i > start - kw_defaults.size(); --i) {
			auto &arg = VirtualMachine::the().stack_local(i);
			if (std::holds_alternative<PyObject *>(arg) && !std::get<PyObject *>(arg)) {
				VirtualMachine::the().stack_local(i) = *kw_default_iter;
			}
			if (auto it = std::find(cell2arg.begin(), cell2arg.end(), i); it != cell2arg.end()) {
				const auto free_var_idx = std::distance(cell2arg.begin(), it);
				auto cell = PyCell::create(*kw_default_iter);
				if (cell.is_err()) return cell;
				function_frame->freevars()[free_var_idx] = cell.unwrap();
			}
			kw_default_iter = std::next(kw_default_iter);
		}
	}

	if (m_code->flags().is_set(CodeFlags::Flag::VARARGS)) {
		std::vector<Value> remaining_args;
		if (args) {
			for (size_t idx = args_count; idx < args->size(); ++idx) {
				remaining_args.push_back(args->elements()[idx]);
			}
		}
		auto args_ = PyTuple::create(remaining_args);
		if (args_.is_err()) { return args_; }
		VirtualMachine::the().stack_local(total_arguments_count) = args_.unwrap();
	} else if (args_count < args->size()) {
		return Err(type_error("{}() takes {} positional arguments but {} given",
			m_name->value(),
			args_count,
			args->size()));
	}

	if (m_code->flags().is_set(CodeFlags::Flag::VARKEYWORDS)) {
		auto remaining_kwargs_ = PyDict::create();
		if (remaining_kwargs_.is_err()) { return remaining_kwargs_; }
		auto *remaining_kwargs = remaining_kwargs_.unwrap();
		if (kwargs) {
			const auto &argnames = m_code->varnames();
			for (const auto &[key, value] : kwargs->map()) {
				auto key_str = std::get<String>(key);
				auto arg_iter = std::find(argnames.begin(), argnames.end(), key_str.s);
				if (arg_iter == argnames.end()) {
					remaining_kwargs->insert(key, value);
					kwargs_count++;
					continue;
				}

				auto &arg =
					VirtualMachine::the().stack_local(std::distance(argnames.begin(), arg_iter));
				if (std::holds_alternative<PyObject *>(arg) && !std::get<PyObject *>(arg)) {
					remaining_kwargs->insert(key, value);
					kwargs_count++;
				}
			}
		}
		size_t kwargs_index = [&]() {
			if (m_code->flags().is_set(CodeFlags::Flag::VARARGS)) {
				return total_arguments_count + 1;
			} else {
				return total_arguments_count;
			}
		}();
		VirtualMachine::the().stack_local(kwargs_index) = remaining_kwargs;
	}

	spdlog::debug("Requesting stack frame with {} virtual registers", m_code->register_count());

	for (size_t idx = m_code->cellvars_count(); const auto &el : m_closure) {
		function_frame->freevars()[idx++] = el;
	}

	// spdlog::debug("Frame: {}", (void *)execution_frame);
	// spdlog::debug("Locals: {}", execution_frame->locals()->to_string());
	// spdlog::debug("Globals: {}", execution_frame->globals()->to_string());
	// if (ns) { spdlog::info("Namespace: {}", ns->to_string()); }
	return VirtualMachine::the().interpreter().call(m_code->function(), function_frame);
}

PyResult<PyObject *> PyFunction::__call__(PyTuple *args, PyDict *kwargs)
{
	auto function_locals = PyDict::create();
	if (function_locals.is_err()) { return function_locals; }
	return call_with_frame(function_locals.unwrap(), args, kwargs);
}

namespace {
	std::once_flag function_flag;
}// namespace

std::unique_ptr<TypePrototype> PyFunction::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(function_flag, []() {
		type = std::move(klass<PyFunction>("function")
							 .attr("__code__", &PyFunction::m_code)
							 .attr("__globals__", &PyFunction::m_globals)
							 .attr("__dict__", &PyFunction::m_dict)
							 .attr("__name__", &PyFunction::m_name)
							 .type);
	});
	return std::move(type);
}


PyNativeFunction::PyNativeFunction(std::string &&name, FunctionType &&function)
	: PyBaseObject(BuiltinTypes::the().native_function()), m_name(std::move(name)),
	  m_function(std::move(function))
{}

std::string PyNativeFunction::to_string() const
{
	return fmt::format("built-in method {} at {}", m_name, (void *)this);
}

PyResult<PyObject *> PyNativeFunction::__call__(PyTuple *args, PyDict *kwargs)
{
	return VirtualMachine::the().interpreter().call(this, args, kwargs);
}

PyResult<PyObject *> PyNativeFunction::__repr__() const { return PyString::create(to_string()); }

void PyNativeFunction::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto *obj : m_captures) { obj->visit_graph(visitor); }
}

PyType *PyNativeFunction::type() const { return native_function(); }

namespace {
	std::once_flag native_function_flag;

	std::unique_ptr<TypePrototype> register_native_function()
	{
		return std::move(klass<PyNativeFunction>("builtin_function_or_method").type);
	}
}// namespace

std::unique_ptr<TypePrototype> PyNativeFunction::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(native_function_flag, []() { type = register_native_function(); });
	return std::move(type);
}

template<> PyFunction *as(PyObject *node)
{
	if (node->type() == function()) { return static_cast<PyFunction *>(node); }
	return nullptr;
}

template<> const PyFunction *as(const PyObject *node)
{
	if (node->type() == function()) { return static_cast<const PyFunction *>(node); }
	return nullptr;
}

template<> PyNativeFunction *as(PyObject *node)
{
	if (node->type() == native_function()) { return static_cast<PyNativeFunction *>(node); }
	return nullptr;
}

template<> const PyNativeFunction *as(const PyObject *node)
{
	if (node->type() == native_function()) { return static_cast<const PyNativeFunction *>(node); }
	return nullptr;
}

}// namespace py