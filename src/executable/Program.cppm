module;

#include "common.hpp"
#include "core.hpp"
#include "executable/CodeFlags.hpp"

export module py.runtime:executable_program;
import :object;
import std;
import py.ast;

export namespace py {
struct Number;
class PyTuple;
}// namespace py

export class VirtualMachine;

export class Program
	: NonCopyable
	, public std::enable_shared_from_this<Program>
{
	std::string m_filename;
	std::vector<std::string> m_argv;

  protected:
	Program() {}

  public:
	Program(std::string &&filename, std::vector<std::string> &&argv);
	virtual ~Program() {}

	virtual int execute(VirtualMachine *) = 0;

	const std::string &filename() const { return m_filename; }
	const std::vector<std::string> &argv() const { return m_argv; }

	void set_filename(std::string filename) { m_filename = std::move(filename); }

	virtual std::string to_string() const = 0;

	virtual py::PyObject *as_pyfunction(const std::string &function_name,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		py::PyTuple *closure) const = 0;

	virtual py::PyObject *main_function() = 0;

	virtual void visit_functions(Cell::Visitor &) const = 0;

	virtual std::vector<std::uint8_t> serialize() const = 0;
};

export namespace compiler {
std::shared_ptr<Program> compile(std::shared_ptr<ast::Module> node,
	std::vector<std::string> argv,
	Backend backend,
	OptimizationLevel lvl);
}
