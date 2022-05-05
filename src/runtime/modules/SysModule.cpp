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
PyResult<PyList *> create_sys_paths(Interpreter &interpreter)
{
	const auto &entry_script = interpreter.entry_script();
	auto entry_parent = PyString::create(std::filesystem::path(entry_script).parent_path());
	if (entry_parent.is_err()) return Err(entry_parent.unwrap_err());
	auto path_list = PyList::create({ entry_parent.unwrap() });

	return path_list;
}

PyResult<PyList *> create_sys_argv(Interpreter &interpreter)
{
	auto argv_list = PyList::create();
	if (argv_list.is_err()) return argv_list;
	for (const auto &arg : interpreter.argv()) {
		auto arg_str = PyString::create(arg);
		if (arg_str.is_err()) { return Err(arg_str.unwrap_err()); }
		argv_list.unwrap()->elements().push_back(arg_str.unwrap());
	}

	return argv_list;
}

}// namespace

namespace py {

PyModule *sys_module(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();

	if (s_sys_module && heap.slab().has_address(bit_cast<uint8_t *>(s_sys_module))) {
		return s_sys_module;
	}

	s_sys_module = heap.allocate<PyModule>(PyString::create("sys").unwrap());
	if (!s_sys_module) { TODO(); }

	s_sys_module->insert(PyString::create("path").unwrap(), create_sys_paths(interpreter).unwrap());
	s_sys_module->insert(PyString::create("argv").unwrap(), create_sys_argv(interpreter).unwrap());

	auto modules = PyDict::create();
	if (modules.is_err()) { TODO(); }
	modules.unwrap()->insert(PyString::create("sys").unwrap(), s_sys_module);
	s_sys_module->insert(PyString::create("modules").unwrap(), modules.unwrap());

	return s_sys_module;
}

}// namespace py