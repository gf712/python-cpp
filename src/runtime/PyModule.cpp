#include "PyModule.hpp"
#include "PyDict.hpp"
#include "PyList.hpp"
#include "modules/Modules.hpp"

#include "executable/Program.hpp"
#include "interpreter/Interpreter.hpp"
#include "parser/Parser.hpp"
#include "vm/VM.hpp"

#include <filesystem>

namespace fs = std::filesystem;

std::optional<std::string> resolve_path(std::string module_name)
{
	auto *sysmodule = sys_module(VirtualMachine::the().interpreter());

	auto *search_paths =
		std::get<PyObject *>(sysmodule->symbol_table().at(PyString::create("path")));

	auto *search_path_list = as<PyList>(search_paths);

	for (const auto &path_value : search_path_list->elements()) {
		const auto path_str = as<PyString>(std::get<PyObject *>(path_value))->value();
		const auto possible_file = fs::path(path_str) / (module_name + ".py");
		spdlog::debug("Checking if file {} exists", possible_file.c_str());
		if (fs::exists(possible_file)) { return possible_file; }
	}

	return {};
}


PyModule::PyModule(PyString *module_name)
	: PyObject(PyObjectType::PY_MODULE), m_module_name(std::move(module_name))
{
	m_attributes.insert_or_assign("__name__", module_name);
}

void PyModule::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_module_name);
	for (auto &[k, v] : m_symbol_table) {
		k->visit_graph(visitor);
		if (std::holds_alternative<PyObject *>(v)) {
			std::get<PyObject *>(v)->visit_graph(visitor);
		}
	}
}

PyModule *PyModule::create(PyString *name)
{
	auto &vm = VirtualMachine::the();
	if (auto *module = vm.interpreter().get_imported_module(name)) { return module; }

	const auto filepath = resolve_path(name->value());
	if (!filepath.has_value()) {
		vm.interpreter().raise_exception(
			"ModuleNotFoundError: No module named '{}'", name->value());
		return nullptr;
	}

	auto lexer = Lexer::create(*filepath);
	parser::Parser p{ lexer };
	p.parse();

	auto program = BytecodeGenerator::compile(p.module(), compiler::OptimizationLevel::None);

	ASSERT(vm.execute(program) == EXIT_SUCCESS);
	auto *module_dict = vm.interpreter().execution_frame()->globals();

	auto *module = vm.heap().allocate<PyModule>(name);

	// hold on to the program until the module is destructed
	// This is important to keep the instruction vector alive
	module->m_program = program;

	for (const auto &[k, v] : module_dict->map()) {
		if (std::holds_alternative<PyObject *>(k)) {
			if (auto pystr = as<PyString>(std::get<PyObject *>(k))) {
				module->m_symbol_table[pystr] = v;
			} else {
				TODO()
			}
		} else if (std::holds_alternative<String>(k)) {
			module->m_symbol_table[PyString::create(std::get<String>(k).s)] = v;
		} else {
			TODO()
		}
	}

	// clean up the interpreter now that we have obtained all the global data we needed
	vm.shutdown_interpreter(vm.interpreter());
	return module;
}
