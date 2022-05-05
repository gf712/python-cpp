#include "PyModule.hpp"
#include "MemoryError.hpp"
#include "PyDict.hpp"
#include "PyList.hpp"
#include "PyString.hpp"
#include "ValueError.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"
#include "interpreter/Interpreter.hpp"
#include "modules/Modules.hpp"
#include "parser/Parser.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

#include <filesystem>

namespace fs = std::filesystem;
using namespace py;

namespace {
std::optional<std::string> resolve_path(std::string module_name)
{
	auto *sysmodule = sys_module(VirtualMachine::the().interpreter());

	ASSERT(sysmodule)

	auto path_pystr_ = PyString::create("path");
	if (path_pystr_.is_err()) return {};
	auto *path_pystr = path_pystr_.unwrap();

	auto *search_paths = std::get<PyObject *>(sysmodule->symbol_table().at(path_pystr));

	auto *search_path_list = as<PyList>(search_paths);

	for (const auto &path_value : search_path_list->elements()) {
		const auto path_str = as<PyString>(std::get<PyObject *>(path_value))->value();
		const auto possible_file = fs::path(path_str) / (module_name + ".py");
		spdlog::debug("Checking if file {} exists", possible_file.c_str());
		if (fs::exists(possible_file)) { return possible_file; }
	}

	return {};
}
}// namespace


PyModule::PyModule(PyString *module_name)
	: PyBaseObject(BuiltinTypes::the().module()), m_module_name(std::move(module_name))
{
	m_attributes = PyDict::create().unwrap();
}

PyResult<PyObject *> PyModule::__repr__() const
{
	return PyString::create(fmt::format("<module '{}'>", m_module_name->to_string()));
}

void PyModule::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_module_name);
	for (auto &[k, v] : m_symbol_table) {
		visitor.visit(*k);
		if (std::holds_alternative<PyObject *>(v)) { visitor.visit(*std::get<PyObject *>(v)); }
	}
}

std::string PyModule::to_string() const
{
	return fmt::format("<module '{}'>", m_module_name->to_string());
}

PyResult<PyModule *> PyModule::create(PyString *name)
{
	auto &vm = VirtualMachine::the();
	auto ip = VirtualMachine::the().instruction_pointer();

	if (auto *module = vm.interpreter().get_imported_module(name)) { return Ok(module); }

	const auto filepath = resolve_path(name->value());
	if (!filepath.has_value()) {
		// FIXME: should throw ModuleNotFoundError, not ValueError
		return Err(value_error("ModuleNotFoundError: No module named '{}'", name->value()));
	}

	auto lexer = Lexer::create(*filepath);
	parser::Parser p{ lexer };
	p.parse();

	auto program =
		codegen::BytecodeGenerator::compile(p.module(), {}, compiler::OptimizationLevel::None);

	auto result = vm.execute(program);
	vm.set_instruction_pointer(ip);
	ASSERT(result == EXIT_SUCCESS);
	auto *module_dict = vm.interpreter().execution_frame()->globals();

	auto *module = vm.heap().allocate<PyModule>(name);
	if (!module) { return Err(memory_error(sizeof(PyModule))); }

	// hold on to the program until the module is destructed
	// This is important to keep the instruction vector alive
	module->m_program = std::move(program);

	for (const auto &[k, v] : module_dict->map()) {
		if (std::holds_alternative<PyObject *>(k)) {
			if (auto pystr = as<PyString>(std::get<PyObject *>(k))) {
				module->m_symbol_table[pystr] = v;
			} else {
				TODO();
			}
		} else if (std::holds_alternative<String>(k)) {
			auto symbol_name = PyString::create(std::get<String>(k).s);
			if (symbol_name.is_err()) { return Err(symbol_name.unwrap_err()); }
			module->m_symbol_table[symbol_name.unwrap()] = v;
		} else {
			TODO();
		}
	}

	// clean up the interpreter now that we have obtained all the global data we needed
	vm.shutdown_interpreter(vm.interpreter());
	return Ok(module);
}

void PyModule::insert(PyString *key, const Value &value)
{
	m_symbol_table.insert_or_assign(key, value);
	auto obj = PyObject::from(value);
	ASSERT(obj.is_ok())
	m_attributes->insert(key, obj.unwrap());
}

PyType *PyModule::type() const { return module(); }

namespace {

std::once_flag module_flag;

std::unique_ptr<TypePrototype> register_module()
{
	return std::move(klass<PyModule>("module").attr("name", &PyModule::m_module_name).type);
}
}// namespace

std::unique_ptr<TypePrototype> PyModule::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(module_flag, []() { type = ::register_module(); });
	return std::move(type);
}

template<> PyModule *py::as(PyObject *obj)
{
	if (obj->type() == module()) { return static_cast<PyModule *>(obj); }
	return nullptr;
}


template<> const PyModule *py::as(const PyObject *obj)
{
	if (obj->type() == module()) { return static_cast<const PyModule *>(obj); }
	return nullptr;
}
