#include "PyCode.hpp"
#include "PyTuple.hpp"
#include "executable/Function.hpp"
#include "executable/bytecode/Bytecode.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "executable/bytecode/serialization/deserialize.hpp"
#include "executable/bytecode/serialization/serialize.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"


namespace py {
PyCode::PyCode(std::unique_ptr<Function> &&function,
	std::vector<std::string> cellvars,
	std::vector<std::string> varnames,
	std::vector<std::string> freevars,
	size_t stack_size,
	std::string filename,
	size_t first_line_number,
	size_t arg_count,
	size_t kwonly_arg_count,
	std::vector<size_t> cell2arg,
	size_t nlocals,
	PyTuple *consts,
	CodeFlags flags)
	: PyBaseObject(BuiltinTypes::the().code()), m_function(std::move(function)),
	  m_register_count(m_function->register_count()), m_cellvars(std::move(cellvars)),
	  m_varnames(std::move(varnames)), m_freevars(std::move(freevars)), m_stack_size(stack_size),
	  m_filename(std::move(filename)), m_first_line_number(first_line_number),
	  m_arg_count(arg_count), m_kwonly_arg_count(kwonly_arg_count), m_cell2arg(std::move(cell2arg)),
	  m_nlocals(nlocals), m_consts(consts), m_flags(flags)
{}

PyCode::~PyCode() {}

size_t PyCode::register_count() const { return m_register_count; }

size_t PyCode::freevars_count() const { return m_freevars.size(); }

size_t PyCode::cellvars_count() const { return m_cellvars.size(); }

const std::vector<size_t> &PyCode::cell2arg() const { return m_cell2arg; }

size_t PyCode::arg_count() const { return m_arg_count; }

size_t PyCode::kwonly_arg_count() const { return m_kwonly_arg_count; }

CodeFlags PyCode::flags() const { return m_flags; }

PyType *PyCode::type() const { return code(); }

const PyTuple *PyCode::consts() const { return m_consts; }

void PyCode::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_consts) { visitor.visit(*const_cast<PyTuple *>(m_consts)); }
}

std::vector<uint8_t> PyCode::serialize() const
{
	std::vector<uint8_t> result;
	auto serialized_function = m_function->serialize();
	::py::serialize(serialized_function.size(), result);
	result.reserve(result.size() + serialized_function.size());
	for (const auto &el : serialized_function) { result.push_back(el); }

	::py::serialize(m_cellvars, result);
	::py::serialize(m_varnames, result);
	::py::serialize(m_freevars, result);
	::py::serialize(m_stack_size, result);
	::py::serialize(m_filename, result);
	::py::serialize(m_first_line_number, result);
	::py::serialize(m_arg_count, result);
	::py::serialize(m_kwonly_arg_count, result);
	::py::serialize(m_cell2arg, result);
	::py::serialize(m_nlocals, result);
	::py::serialize(m_consts, result);
	::py::serialize(static_cast<uint8_t>(m_flags.bits().to_ulong()), result);

	return result;
}

std::pair<PyCode *, size_t> PyCode::deserialize(std::span<const uint8_t> &buffer)
{
	std::cout << "deserialize " << buffer.size() << '\n';
	size_t serialized_function_size{ 0 };
	for (size_t i = 0; i < sizeof(size_t); ++i) {
		reinterpret_cast<uint8_t *>(&serialized_function_size)[i] = buffer[i];
	}

	buffer = buffer.subspan(sizeof(size_t), buffer.size() - sizeof(size_t));

	auto function = Bytecode::deserialize(buffer);
	const auto cellvars = ::py::deserialize<std::vector<std::string>>(buffer);
	const auto varnames = ::py::deserialize<std::vector<std::string>>(buffer);
	const auto freevars = ::py::deserialize<std::vector<std::string>>(buffer);
	const auto stack_size = ::py::deserialize<size_t>(buffer);
	const auto filename = ::py::deserialize<std::string>(buffer);
	const auto first_line_number = ::py::deserialize<size_t>(buffer);
	const auto arg_count = ::py::deserialize<size_t>(buffer);
	const auto kwonly_arg_count = ::py::deserialize<size_t>(buffer);
	const auto cell2arg = ::py::deserialize<std::vector<size_t>>(buffer);
	const auto nlocals = ::py::deserialize<size_t>(buffer);
	const auto consts = ::py::deserialize<PyTuple *>(buffer);
	const auto flags = ::py::deserialize<uint8_t>(buffer);

	return { VirtualMachine::the().heap().allocate<PyCode>(std::move(function),
				 cellvars,
				 varnames,
				 freevars,
				 stack_size,
				 filename,
				 first_line_number,
				 arg_count,
				 kwonly_arg_count,
				 cell2arg,
				 nlocals,
				 consts,
				 CodeFlags::from_byte(flags)),
		0 };
}


namespace {
	std::once_flag code_flag;

	std::unique_ptr<TypePrototype> register_code() { return std::move(klass<PyCode>("code").type); }
}// namespace

std::unique_ptr<TypePrototype> PyCode::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(code_flag, []() { type = register_code(); });
	return std::move(type);
}
}// namespace py