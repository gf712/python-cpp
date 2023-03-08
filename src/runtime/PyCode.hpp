#include "PyObject.hpp"
#include "executable/Program.hpp"
#include <span>

namespace py {

class PyCode : public PyBaseObject
{
  public:
	// implementation details
	const std::unique_ptr<Function> m_function;
	const size_t m_register_count;
	const std::vector<size_t> m_cell2arg;

	// code object public attributes
	// number of arguments (not including keyword only arguments, * or ** args)
	const size_t m_arg_count;
	// string of raw compiled bytecode
	// const std::string m_code;
	// tuple of names of cell variables (referenced by containing scopes)
	const std::vector<std::string> m_cellvars;
	// tuple of constants used in the bytecode
	PyTuple *m_consts{ nullptr };
	// name of file in which this code object was created
	const std::string m_filename;
	// number of first line in Python source code
	const size_t m_first_line_number;
	// bitmap of CO_* flags
	CodeFlags m_flags;
	// encoded mapping of line numbers to bytecode indices
	PyDict *m_lnotab{ nullptr };
	// tuple of names of free variables (referenced via a function’s closure)
	const std::vector<std::string> m_freevars;
	// number of positional only arguments
	const size_t m_positional_only_arg_count;
	// number of keyword only arguments (not including ** arg)
	const size_t m_kwonly_arg_count;
	// name with which this code object was defined
	const std::string m_name;
	// tuple of names other than arguments and function locals
	const std::vector<std::string> m_names;
	// number of local variables
	const size_t m_nlocals;
	// virtual machine stack space required
	const size_t m_stack_size;
	// tuple of names of arguments and local variables
	const std::vector<std::string> m_varnames;

	std::shared_ptr<Program> m_program;

	PyCode(PyType *);

	PyCode(std::unique_ptr<Function> &&function,
		std::vector<size_t> &&cell2arg,
		size_t arg_count,
		std::vector<std::string> &&cellvars,
		PyTuple *consts,
		std::string &&filename,
		size_t first_line_number,
		CodeFlags flags,
		std::vector<std::string> &&freevars,
		size_t positional_arg_count,
		size_t kwonly_arg_count,
		size_t stack_size,
		std::string &&name,
		std::vector<std::string> &&names,
		size_t nlocals,
		std::vector<std::string> &&varnames);

  public:
	~PyCode();

	static PyResult<PyCode *> create(std::unique_ptr<Function> &&function,
		std::vector<size_t> cell2arg,
		size_t arg_count,
		std::vector<std::string> cellvars,
		PyTuple *consts,
		std::string filename,
		size_t first_line_number,
		CodeFlags flags,
		std::vector<std::string> freevars,
		size_t positional_arg_count,
		size_t kwonly_arg_count,
		size_t stack_size,
		std::string name,
		std::vector<std::string> names,
		size_t nlocals,
		std::vector<std::string> varnames);

	static PyResult<PyCode *> create(std::shared_ptr<Program> program);

	PyObject *call(PyTuple *args, PyDict *kwargs);
	const std::vector<std::string> &varnames() const { return m_varnames; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;

	size_t register_count() const;
	size_t freevars_count() const;
	size_t cellvars_count() const;
	size_t arg_count() const;
	size_t kwonly_arg_count() const;
	CodeFlags flags() const;
	const std::vector<size_t> &cell2arg() const;
	const PyTuple *consts() const;

	const std::string &name() const { return m_name; }
	const std::vector<std::string> &names() const;

	const std::unique_ptr<Function> &function() const { return m_function; }

	const std::shared_ptr<Program> &program() const { return m_program; }

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	void visit_graph(Visitor &) override;

	std::vector<uint8_t> serialize() const;

	PyResult<PyObject *> eval(PyDict *globals,
		PyDict *locals,
		PyTuple *args,
		PyDict *kwargs,
		const std::vector<Value> &defaults,
		const std::vector<Value> &kw_defaults,
		const std::vector<Value> &closure,
		PyString *name) const;

	PyObject *make_function(const std::string &function_name,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		PyTuple *closure) const;

	static std::pair<PyResult<PyCode *>, size_t> deserialize(std::span<const uint8_t> &,
		std::shared_ptr<Program>);
};

}// namespace py
