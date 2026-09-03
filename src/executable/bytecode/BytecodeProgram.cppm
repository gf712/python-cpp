export module py.runtime:bytecode_program;
import :bytecode;
import :code;
import :object;
import :tuple;
import :value;
import :executable_program;
import std;

export class VirtualMachine;


export class BytecodeProgram : public Program
{
	std::vector<py::PyCode *> m_functions;
	py::PyCode *m_main_function;
	std::vector<std::shared_ptr<Program>> m_backends;

	BytecodeProgram() {}

	BytecodeProgram(std::string filename, std::vector<std::string> argv);

  public:
	static std::shared_ptr<BytecodeProgram>
		create(FunctionBlocks &&func_blocks, std::string filename, std::vector<std::string> argv);

	InstructionVector::const_iterator begin() const;

	InstructionVector::const_iterator end() const;

	std::size_t main_stack_size() const;

	py::PyObject *as_pyfunction(const std::string &function_name,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		py::PyTuple *closure) const override;

	std::string to_string() const override;

	int execute(VirtualMachine *) override;

	const auto &functions() const { return m_functions; }

	py::PyObject *main_function() override;

	void add_backend(std::shared_ptr<Program>);

	void visit_functions(Cell::Visitor &) const override;

	std::vector<std::uint8_t> serialize() const final;

	static std::shared_ptr<BytecodeProgram> deserialize(const std::vector<std::uint8_t> &);
};
