#include "PyFunction.hpp"
#include "PyBoundMethod.hpp"
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

using namespace py;

PyCode::PyCode(std::shared_ptr<Function> function,
	std::vector<std::string> varnames,
	std::vector<Value> defaults,
	std::vector<Value> kwonly_defaults,
	size_t arg_count,
	size_t kwonly_arg_count,
	CodeFlags flags,
	PyModule *module)
	: PyBaseObject(BuiltinTypes::the().code()), m_function(function),
	  m_register_count(function->registers_needed()), m_varnames(std::move(varnames)),
	  m_defaults(std::move(defaults)), m_kwonly_defaults(std::move(kwonly_defaults)),
	  m_arg_count(arg_count), m_kwonly_arg_count(kwonly_arg_count), m_flags(flags), m_module(module)
{}

size_t PyCode::register_count() const { return m_register_count; }

size_t PyCode::arg_count() const { return m_arg_count; }

size_t PyCode::kwonly_arg_count() const { return m_kwonly_arg_count; }

CodeFlags PyCode::flags() const { return m_flags; }

void PyCode::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	// FIXME: this should probably never be null
	if (m_module) m_module->visit_graph(visitor);
}

PyType *PyCode::type() const { return code(); }

namespace {

std::once_flag code_flag;

std::unique_ptr<TypePrototype> register_code() { return std::move(klass<PyCode>("code").type); }
}// namespace

std::unique_ptr<TypePrototype> PyCode::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(code_flag, []() { type = ::register_code(); });
	return std::move(type);
}

PyFunction::PyFunction(std::string name, PyCode *code, PyDict *globals)
	: PyBaseObject(BuiltinTypes::the().function()), m_name(std::move(name)), m_code(code),
	  m_globals(globals)
{}

void PyFunction::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	m_code->visit_graph(visitor);
	if (m_globals) visitor.visit(*m_globals);
}

PyType *PyFunction::type() const { return function(); }

PyObject *PyFunction::__repr__() const
{
	return PyString::create(fmt::format("<function {}>", m_name));
}

PyObject *PyFunction::__get__(PyObject *instance, PyObject * /*owner*/) const
{
	if (!instance || instance == py_none()) { return const_cast<PyFunction *>(this); }
	return PyBoundMethod::create(instance, const_cast<PyFunction *>(this));
}

PyObject *PyFunction::call_with_frame(PyDict *locals, PyTuple *args, PyDict *kwargs) const
{
	auto *function_frame =
		ExecutionFrame::create(VirtualMachine::the().interpreter().execution_frame(),
			m_code->register_count(),
			m_globals,
			locals);

	const auto &varnames = m_code->varnames();
	std::vector<std::string> positional_args{ varnames.begin(),
		varnames.begin() + m_code->arg_count() };
	std::vector<std::string> keyword_only_args{ varnames.begin() + m_code->arg_count(),
		varnames.end() };

	size_t args_count = 0;
	size_t kwargs_count = 0;

	if (args) {
		size_t max_args = std::min(args->size(), m_code->arg_count());
		for (size_t idx = 0; idx < max_args; ++idx) {
			function_frame->parameter(idx) = args->elements()[idx];
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
					VirtualMachine::the().interpreter().raise_exception(type_error(
						"{}() got an unexpected keyword argument '{}'", m_name, key_str.s));
					return nullptr;
				}
			}
			auto &arg = function_frame->parameter(std::distance(argnames.begin(), arg_iter));
			if (arg.has_value()) {
				VirtualMachine::the().interpreter().raise_exception(
					type_error("{}() got multiple values for argument '{}'", m_name, key_str.s));
				return nullptr;
			}
			arg = value;
			kwargs_count++;
		}
	}

	{
		const auto &defaults = m_code->defaults();
		auto default_iter = defaults.rbegin();
		for (size_t i = m_code->arg_count() - 1; i > (m_code->arg_count() - defaults.size() - 1);
			 --i) {
			if (!function_frame->parameter(i).has_value()) {
				function_frame->parameter(i) = *default_iter;
			}
			default_iter = std::next(default_iter);
		}

		const auto &kw_defaults = m_code->kwonly_defaults();
		auto kw_default_iter = kw_defaults.rbegin();
		const size_t start = m_code->kwonly_arg_count() + m_code->arg_count() - 1;
		for (size_t i = start; i > start - kw_defaults.size(); --i) {
			if (!function_frame->parameter(i).has_value()) {
				function_frame->parameter(i) = *kw_default_iter;
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
		function_frame->parameter(m_code->varnames().size()) = PyTuple::create(remaining_args);
	} else if (args_count < args->size()) {
		VirtualMachine::the().interpreter().raise_exception(type_error(
			"{}() takes {} positional arguments but {} given", m_name, args_count, args->size()));
		return nullptr;
	}

	if (m_code->flags().is_set(CodeFlags::Flag::VARKEYWORDS)) {
		auto *remaining_kwargs = PyDict::create();
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

				auto &arg = function_frame->parameter(std::distance(argnames.begin(), arg_iter));
				if (!arg.has_value()) {
					remaining_kwargs->insert(key, value);
					kwargs_count++;
				}
			}
		}
		size_t kwargs_index = [&]() {
			if (m_code->flags().is_set(CodeFlags::Flag::VARARGS)) {
				return m_code->varnames().size() + 1;
			} else {
				return m_code->varnames().size();
			}
		}();
		function_frame->parameter(kwargs_index) = remaining_kwargs;
	}

	spdlog::debug("Requesting stack frame with {} virtual registers", m_code->register_count());

	// spdlog::debug("Frame: {}", (void *)execution_frame);
	// spdlog::debug("Locals: {}", execution_frame->locals()->to_string());
	// spdlog::debug("Globals: {}", execution_frame->globals()->to_string());
	// if (ns) { spdlog::info("Namespace: {}", ns->to_string()); }
	return VirtualMachine::the().interpreter().call(m_code->function(), function_frame);
}

PyObject *PyFunction::__call__(PyTuple *args, PyDict *kwargs)
{
	auto function_locals = VirtualMachine::the().heap().allocate<PyDict>();
	return call_with_frame(function_locals, args, kwargs);
}

namespace {

std::once_flag function_flag;

std::unique_ptr<TypePrototype> register_function()
{
	return std::move(klass<PyFunction>("function").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyFunction::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(function_flag, []() { type = ::register_function(); });
	return std::move(type);
}


PyNativeFunction::PyNativeFunction(std::string &&name,
	std::function<PyObject *(PyTuple *, PyDict *)> &&function)
	: PyBaseObject(BuiltinTypes::the().native_function()), m_name(std::move(name)),
	  m_function(std::move(function))
{}

PyObject *PyNativeFunction::__call__(PyTuple *args, PyDict *kwargs)
{
	return VirtualMachine::the().interpreter().call(this, args, kwargs);
}

PyObject *PyNativeFunction::__repr__() const
{
	return PyString::create(fmt::format("built-in method {} at {}", m_name, (void *)this));
}

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
	std::call_once(native_function_flag, []() { type = ::register_native_function(); });
	return std::move(type);
}

template<> PyFunction *py::as(PyObject *node)
{
	if (node->type() == function()) { return static_cast<PyFunction *>(node); }
	return nullptr;
}

template<> const PyFunction *py::as(const PyObject *node)
{
	if (node->type() == function()) { return static_cast<const PyFunction *>(node); }
	return nullptr;
}

template<> PyNativeFunction *py::as(PyObject *node)
{
	if (node->type() == native_function()) { return static_cast<PyNativeFunction *>(node); }
	return nullptr;
}

template<> const PyNativeFunction *py::as(const PyObject *node)
{
	if (node->type() == native_function()) { return static_cast<const PyNativeFunction *>(node); }
	return nullptr;
}