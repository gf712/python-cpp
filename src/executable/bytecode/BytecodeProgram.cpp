#include "BytecodeProgram.hpp"
#include "Bytecode.hpp"
#include "executable/Function.hpp"
#include "executable/Mangler.hpp"
#include "interpreter/InterpreterSession.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyTraceback.hpp"
#include "runtime/PyTuple.hpp"

#include <numeric>

using namespace py;

BytecodeProgram::BytecodeProgram(std::string filename, std::vector<std::string> argv)
	: Program(std::move(filename), std::move(argv))
{}

std::shared_ptr<BytecodeProgram> BytecodeProgram::create(FunctionBlocks &&func_blocks,
	std::string filename,
	std::vector<std::string> argv)
{
	auto program = std::shared_ptr<BytecodeProgram>(new BytecodeProgram{ filename, argv });

	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();

	std::vector<size_t> functions_instruction_count;
	functions_instruction_count.reserve(func_blocks.functions.size());
	for (const auto &f : func_blocks.functions) {
		functions_instruction_count.push_back(std::transform_reduce(
			f.blocks.begin(), f.blocks.end(), 0u, std::plus<size_t>{}, [](const auto &ins) {
				return ins.size();
			}));
	}

	// have to reserve instruction vector to avoid relocations
	// since the iterators depend on the vector memory layout
	InstructionVector main_instructions;
	main_instructions.reserve(functions_instruction_count[0]);

	auto &main_func = func_blocks.functions.front();

	std::vector<View> main_blocks;
	main_blocks.reserve(main_func.blocks.size());

	for (size_t start_idx = 0; auto &block : main_func.blocks) {
		// ASSERT(!block.empty())
		if (block.empty()) { continue; }
		for (auto &ins : block) { main_instructions.push_back(std::move(ins)); }
		InstructionVector::const_iterator start = main_instructions.cbegin() + start_idx;
		InstructionVector::const_iterator end = main_instructions.end();
		main_blocks.emplace_back(start, end);
		start_idx = main_instructions.size();
	}

	auto main_bytecode = std::make_unique<Bytecode>(main_func.metadata.register_count,
		main_func.metadata.stack_size,
		main_func.metadata.function_name,
		std::move(main_instructions),
		main_blocks,
		program);
	auto consts = PyTuple::create(main_func.metadata.consts);
	if (consts.is_err()) { TODO(); }
	auto main_function = PyCode::create(std::move(main_bytecode),
		main_func.metadata.cell2arg,
		main_func.metadata.arg_count,
		main_func.metadata.cellvars,
		consts.unwrap(),
		main_func.metadata.filename,
		main_func.metadata.first_line_number,
		main_func.metadata.flags,
		main_func.metadata.freevars,
		main_func.metadata.positional_arg_count,
		main_func.metadata.kwonly_arg_count,
		main_func.metadata.stack_size,
		main_func.metadata.function_name,
		main_func.metadata.names,
		main_func.metadata.nlocals,
		main_func.metadata.varnames);

	if (main_function.is_err()) { TODO(); }

	program->m_main_function = main_function.unwrap();

	for (size_t i = 1; i < func_blocks.functions.size(); ++i) {
		auto &func = *std::next(func_blocks.functions.begin(), i);
		std::vector<View> func_blocks_view;
		InstructionVector func_instructions;
		func_instructions.reserve(functions_instruction_count[i]);
		for (size_t start_idx = 0; auto &block : func.blocks) {
			// ASSERT(!block.empty())
			if (block.empty()) { continue; }
			for (auto &ins : block) { func_instructions.push_back(std::move(ins)); }
			InstructionVector::const_iterator start = func_instructions.cbegin() + start_idx;
			InstructionVector::const_iterator end = func_instructions.end();
			func_blocks_view.emplace_back(start, end);
			start_idx = func_instructions.size();
		}

		auto bytecode = std::make_unique<Bytecode>(func.metadata.register_count,
			func.metadata.stack_size,
			func.metadata.function_name,
			std::move(func_instructions),
			func_blocks_view,
			program);
		consts = PyTuple::create(func.metadata.consts);
		if (consts.is_err()) { TODO(); }
		auto code = PyCode::create(std::move(bytecode),
			func.metadata.cell2arg,
			func.metadata.arg_count,
			func.metadata.cellvars,
			consts.unwrap(),
			func.metadata.filename,
			func.metadata.first_line_number,
			func.metadata.flags,
			func.metadata.freevars,
			func.metadata.positional_arg_count,
			func.metadata.kwonly_arg_count,
			func.metadata.stack_size,
			func.metadata.function_name,
			func.metadata.names,
			func.metadata.nlocals,
			func.metadata.varnames);

		if (code.is_err()) { TODO(); }

		program->m_functions.emplace_back(code.unwrap());
	}

	return program;
}

size_t BytecodeProgram::main_stack_size() const { return m_main_function->register_count(); }

std::vector<View>::const_iterator BytecodeProgram::begin() const
{
	// FIXME: assumes all functions are bytecode
	ASSERT(m_main_function->function()->backend() == FunctionExecutionBackend::BYTECODE)
	return static_cast<Bytecode *>(m_main_function->function().get())->begin();
}

std::vector<View>::const_iterator BytecodeProgram::end() const
{
	// FIXME: assumes all functions are bytecode
	ASSERT(m_main_function->function()->backend() == FunctionExecutionBackend::BYTECODE)
	return static_cast<Bytecode *>(m_main_function->function().get())->end();
}

std::string BytecodeProgram::to_string() const
{
	std::stringstream ss;
	for (const auto &func : m_functions) {
		ss << func->function()->function_name() << ":\n";
		ss << func->function()->to_string() << '\n';
	}

	ss << "main:\n";
	ss << m_main_function->function()->to_string() << '\n';
	return ss.str();
}

int BytecodeProgram::execute(VirtualMachine *vm)
{
	auto &interpreter = vm->initialize_interpreter(shared_from_this());

	auto result = m_main_function->function()->call(*vm, interpreter);

	if (result.is_err()) {
		auto *exception = interpreter.execution_frame()->pop_exception();
		ASSERT(exception == result.unwrap_err())
		std::cout << exception->format_traceback() << std::endl;

		// if (interpreter.execution_frame()->exception_info().has_value()) {
		// 	std::cout << "During handling of the above exception, another exception occurred:\n\n";
		// 	exception = interpreter.execution_frame()->pop_exception();
		// 	std::cout << exception->format_traceback() << std::endl;
		// 	if (interpreter.execution_frame()->exception_info().has_value()) {
		// 		// how many exceptions is one meant to expect? :(
		// 		TODO();
		// 	}
		// }
	}

	return result.is_ok() ? EXIT_SUCCESS : EXIT_FAILURE;
}

PyObject *BytecodeProgram::as_pyfunction(const std::string &function_name,
	const std::vector<Value> &default_values,
	const std::vector<Value> &kw_default_values,
	const std::vector<PyCell *> &closure) const
{
	for (const auto &backend : m_backends) {
		if (auto *f =
				backend->as_pyfunction(function_name, default_values, kw_default_values, closure)) {
			return f;
		}
	}
	if (auto it = std::find_if(m_functions.begin(),
			m_functions.end(),
			[&function_name](const auto &f) {
				ASSERT(f->function());
				return f->function()->function_name() == function_name;
			});
		it != m_functions.end()) {
		auto *code = *it;
		const auto &demangled_name = Mangler::default_mangler().function_demangle(function_name);
		return VirtualMachine::the().heap().allocate<PyFunction>(demangled_name,
			default_values,
			kw_default_values,
			code,
			closure,
			VirtualMachine::the().interpreter().execution_frame()->globals());
	}
	return nullptr;
}

PyObject *BytecodeProgram::main_function() { return m_main_function; }

void BytecodeProgram::add_backend(std::shared_ptr<Program> other)
{
	m_backends.push_back(std::move(other));
}

std::string FunctionBlock::to_string() const
{
	std::ostringstream os;
	os << "Function name: " << metadata.function_name << '\n';
	size_t block_idx{ 0 };
	for (const auto &block : blocks) {
		os << "  block " << block_idx++ << '\n';
		for (const auto &ins : block) { os << "    " << ins->to_string() << '\n'; }
	}
	return os.str();
}

void BytecodeProgram::visit_functions(Cell::Visitor &visitor) const
{
	visitor.visit(*const_cast<PyCode *>(m_main_function));
	for (auto &f : m_functions) { visitor.visit(*const_cast<PyCode *>(f)); };
}

std::vector<uint8_t> BytecodeProgram::serialize() const
{
	std::vector<uint8_t> result;
	const auto main_func_serialized = m_main_function->serialize();
	result.insert(result.end(), main_func_serialized.begin(), main_func_serialized.end());

	for (const auto &func : m_functions) {
		const auto func_serialized = func->serialize();
		result.insert(result.end(), func_serialized.begin(), func_serialized.end());
	}

	// TODO: Add support to serialize functions from different backends
	ASSERT(m_backends.empty())

	return result;
}

std::shared_ptr<BytecodeProgram> BytecodeProgram::deserialize(const std::vector<uint8_t> &buffer)
{
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();
	auto program = std::shared_ptr<BytecodeProgram>(new BytecodeProgram);

	auto span = std::span{ buffer };
	auto deserialized_result = PyCode::deserialize(span, program);
	ASSERT(deserialized_result.first.is_ok())
	program->m_main_function = deserialized_result.first.unwrap();
	spdlog::debug(
		"Deserialized main function:\n{}\n\n", program->m_main_function->function()->to_string());

	while (!span.empty()) {
		deserialized_result = PyCode::deserialize(span, program);
		ASSERT(deserialized_result.first.is_ok())
		program->m_functions.push_back(deserialized_result.first.unwrap());
		spdlog::debug("Deserialized function {}:\n{}\n\n",
			program->m_functions.back()->function()->function_name(),
			program->m_functions.back()->function()->to_string());
	}

	return program;
}
