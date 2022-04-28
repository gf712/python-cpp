#include "runtime/PyDict.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"

#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

#include <filesystem>

using namespace py;

static PyModule *s_sys_module = nullptr;

namespace {
PyResult create_sys_paths(Interpreter &interpreter)
{
	const auto &entry_script = interpreter.entry_script();
	auto entry_parent = PyString::create(std::filesystem::path(entry_script).parent_path());
	if (entry_parent.is_err()) return entry_parent;
	auto path_list = PyList::create({ entry_parent.unwrap_as<PyString>() });

	return path_list;
}

PyResult create_sys_argv(Interpreter &interpreter)
{
	auto argv_list_ = PyList::create();
	if (argv_list_.is_err()) return argv_list_;
	auto *argv_list = argv_list_.unwrap_as<PyList>();
	for (const auto &arg : interpreter.argv()) {
		auto arg_str = PyString::create(arg);
		if (arg_str.is_err()) { return arg_str; }
		argv_list->elements().push_back(arg_str.unwrap_as<PyString>());
	}

	return argv_list_;
}

}// namespace

namespace py {

PyModule *sys_module(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();

	if (s_sys_module && heap.slab().has_address(bit_cast<uint8_t *>(s_sys_module))) {
		return s_sys_module;
	}

	s_sys_module = heap.allocate<PyModule>(PyString::create("sys").unwrap_as<PyString>());
	if (!s_sys_module) { TODO(); }

	s_sys_module->insert(
		PyString::create("path").unwrap_as<PyString>(), create_sys_paths(interpreter).unwrap());
	s_sys_module->insert(
		PyString::create("argv").unwrap_as<PyString>(), create_sys_argv(interpreter).unwrap());

	auto modules_ = PyDict::create();
	if (modules_.is_err()) { TODO(); }
	auto *modules = modules_.unwrap_as<PyDict>();
	modules->insert(PyString::create("sys").unwrap_as<PyString>(), s_sys_module);
	s_sys_module->insert(PyString::create("modules").unwrap_as<PyString>(), modules);

	return s_sys_module;
}

}// namespace py