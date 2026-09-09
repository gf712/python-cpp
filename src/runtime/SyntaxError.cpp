module;
#include "core.hpp"

#include <cstdint>

module py.runtime;
import py.types;


namespace py {

namespace {
	constexpr std::string_view whitespace = " \t\n\v\f\r";

	std::string_view strip(std::string_view s)
	{
		const auto first = s.find_first_not_of(whitespace);
		if (first == std::string_view::npos) { return {}; }
		return s.substr(first, s.find_last_not_of(whitespace) - first + 1);
	}

	// `as<T>` dereferences its argument, and every location attribute is null
	// until an info tuple sets it.
	template<typename T> const T *attribute_as(PyObject *attribute)
	{
		return attribute ? as<T>(attribute) : nullptr;
	}
}// namespace

SyntaxError *SyntaxError::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<SyntaxError>(args);
}

SyntaxError *SyntaxError::create(std::string message)
{
	auto args = PyTuple::create(String{ std::move(message) });
	if (args.is_err()) { TODO(); }
	auto &heap = VirtualMachine::the().heap();
	auto *error = heap.allocate<SyntaxError>(args.unwrap());
	auto result = error->__init__(args.unwrap(), nullptr);
	ASSERT(result.is_ok());
	return error;
}

SyntaxError *SyntaxError::create(std::string message, SyntaxErrorLocation location)
{
	auto info = PyTuple::create(String{ std::move(location.filename) },
		Number{ static_cast<int64_t>(location.lineno) },
		Number{ static_cast<int64_t>(location.offset) },
		String{ std::move(location.text) });
	if (info.is_err()) { TODO(); }
	auto args = PyTuple::create(String{ std::move(message) }, info.unwrap());
	if (args.is_err()) { TODO(); }
	auto &heap = VirtualMachine::the().heap();
	auto *error = heap.allocate<SyntaxError>(args.unwrap());
	auto result = error->__init__(args.unwrap(), nullptr);
	ASSERT(result.is_ok());
	return error;
}

SyntaxError::SyntaxError(PyType *type) : Exception(type) {}

SyntaxError::SyntaxError(PyTuple *args) : Exception(types::BuiltinTypes::the().syntax_error(), args)
{}

PyResult<PyObject *> SyntaxError::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::syntax_error());
	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("SyntaxError() takes no keyword arguments"));
	}
	return Ok(SyntaxError::create(args));
}

PyResult<int32_t> SyntaxError::__init__(PyTuple *args, PyDict *kwargs)
{
	// `SyntaxError(msg, (filename, lineno, offset, text))`
	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("SyntaxError() takes no keyword arguments"));
	}
	m_args = args;
	if (!args || args->size() == 0) { return Ok(1); }
	auto msg = PyObject::from(args->elements()[0]);
	if (msg.is_err()) { return Err(msg.unwrap_err()); }
	m_msg = msg.unwrap();
	if (args->size() == 2) {
		auto info = PyObject::from(args->elements()[1]);
		if (info.is_err()) { return Err(info.unwrap_err()); }
		std::vector<Value> info_args;
		info_args.reserve(4);
		if (auto result = from_iterable(info.unwrap(), std::inserter(info_args, info_args.begin()));
			result.is_err()) {
			return Err(result.unwrap_err());
		}
		if (info_args.size() != 4) { return Err(index_error("tuple index out of range")); }
		m_filename = PyObject::from(info_args[0]).unwrap();
		m_lineno = PyObject::from(info_args[1]).unwrap();
		m_offset = PyObject::from(info_args[2]).unwrap();
		m_text = PyObject::from(info_args[3]).unwrap();
	}
	return Ok(1);
}

PyResult<PyObject *> SyntaxError::__str__() const
{
	std::string msg{ "None" };
	if (m_msg) {
		auto str = m_msg->str();
		if (str.is_err()) { return str; }
		msg = str.unwrap()->to_string();
	}
	const auto *filename = attribute_as<PyString>(m_filename);
	const auto *lineno = attribute_as<PyInteger>(m_lineno);
	if (!filename && !lineno) { return PyString::create(msg); }
	std::string basename;
	if (filename) {
		const auto &value = filename->value();
		const auto separator = value.find_last_of("/\\");
		basename = separator == std::string::npos ? value : value.substr(separator + 1);
	}
	if (filename && lineno) {
		return PyString::create(
			std::format("{} ({}, line {})", msg, basename, lineno->as_size_t()));
	}
	if (filename) { return PyString::create(std::format("{} ({})", msg, basename)); }
	return PyString::create(std::format("{} (line {})", msg, lineno->as_size_t()));
}

std::string SyntaxError::format_exception_only() const
{
	std::ostringstream out;

	const auto *lineno = attribute_as<PyInteger>(m_lineno);
	if (lineno) {
		const auto *filename = attribute_as<PyString>(m_filename);
		out << std::format("  File \"{}\", line {}\n",
			filename ? filename->value() : std::string{ "<string>" },
			lineno->as_size_t());
		if (const auto *text = attribute_as<PyString>(m_text)) {
			const std::string_view line{ text->value() };
			if (const auto trimmed = strip(line); !trimmed.empty()) {
				out << "    " << trimmed << "\n";
				if (const auto *offset = attribute_as<PyInteger>(m_offset)) {
					const auto column = std::min(line.size(), offset->as_size_t());
					auto prefix = line.substr(0, column > 0 ? column - 1 : 0);
					if (const auto first = prefix.find_first_not_of(whitespace);
						first != std::string_view::npos) {
						prefix.remove_prefix(first);
					} else {
						prefix = {};
					}
					std::string caret;
					caret.reserve(prefix.size() + 1);
					for (const char c : prefix) {
						caret += whitespace.find(c) != std::string_view::npos ? c : ' ';
					}
					caret += '^';
					out << "    " << caret << "\n";
				}
			}
		}
	}

	std::string msg{ "<no detail available>" };
	if (m_msg && m_msg != py_none()) {
		if (auto str = m_msg->str(); str.is_ok()) { msg = str.unwrap()->to_string(); }
	}
	out << std::format("{}: {}\n", type()->name(), msg);
	return out.str();
}

PyType *SyntaxError::static_type() const
{
	ASSERT(types::syntax_error());
	return types::syntax_error();
}

void SyntaxError::visit_graph(Visitor &visitor)
{
	Exception::visit_graph(visitor);
	if (m_msg) visitor.visit(*m_msg);
	if (m_filename) visitor.visit(*m_filename);
	if (m_text) visitor.visit(*m_text);
	if (m_lineno) visitor.visit(*m_lineno);
	if (m_offset) visitor.visit(*m_offset);
}

namespace {

	std::once_flag syntax_error_flag;

	std::unique_ptr<TypePrototype> register_syntax_error()
	{
		return std::move(klass<SyntaxError>("SyntaxError", Exception::class_type())
				.attr("msg", &SyntaxError::m_msg)
				.attr("filename", &SyntaxError::m_filename)
				.attr("lineno", &SyntaxError::m_lineno)
				.attr("offset", &SyntaxError::m_offset)
				.attr("text", &SyntaxError::m_text)
				.type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> SyntaxError::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(syntax_error_flag, []() { type = register_syntax_error(); });
		return std::move(type);
	};
}

}// namespace py
