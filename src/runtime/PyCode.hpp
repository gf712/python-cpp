#include "PyObject.hpp"
#include "executable/Program.hpp"
#include <span>

namespace py {

class PyCode : public PyBaseObject
{
  public:
	const std::unique_ptr<Function> m_function;
	const size_t m_register_count;

	const std::vector<std::string> m_cellvars;
	const std::vector<std::string> m_varnames;
	const std::vector<std::string> m_freevars;
	const size_t m_stack_size;
	const std::string m_filename;
	const size_t m_first_line_number;
	const size_t m_arg_count;
	const size_t m_kwonly_arg_count;
	const std::vector<size_t> m_cell2arg;
	const size_t m_nlocals;
	PyTuple *m_consts = nullptr;
	CodeFlags m_flags;

  public:
	PyCode(std::unique_ptr<Function> &&function,
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
		CodeFlags flags);

	~PyCode();

	PyObject *call(PyTuple *args, PyDict *kwargs);
	const std::vector<std::string> &varnames() const { return m_varnames; }

	std::string to_string() const override { return fmt::format("PyCode"); }

	size_t register_count() const;
	size_t freevars_count() const;
	size_t cellvars_count() const;
	size_t arg_count() const;
	size_t kwonly_arg_count() const;
	CodeFlags flags() const;
	const std::vector<size_t> &cell2arg() const;
	const PyTuple *consts() const;

	const std::unique_ptr<Function> &function() const { return m_function; }

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

	void visit_graph(Visitor &) override;

	std::vector<uint8_t> serialize() const;

	static std::pair<PyCode *, size_t> deserialize(std::span<const uint8_t> &);
};

}// namespace py