#include "runtime/PyList.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"

#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

#include <filesystem>

using namespace py;

static PyModule *s_sys_module = nullptr;

namespace {
PyList *create_sys_paths(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();

	auto *path_list = heap.allocate<PyList>();
	const auto &entry_script = interpreter.entry_script();
	path_list->elements().push_back(
		PyString::create(std::filesystem::path(entry_script).parent_path()));

	return path_list;
}

PyList *create_sys_argv(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();
	auto *argv_list = heap.allocate<PyList>();
	for (const auto &arg : interpreter.argv()) {
		argv_list->elements().push_back(PyString::create(arg));
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

	s_sys_module = heap.allocate<PyModule>(PyString::create("sys"));

	s_sys_module->insert(PyString::create("path"), create_sys_paths(interpreter));
	s_sys_module->insert(PyString::create("argv"), create_sys_argv(interpreter));

	return s_sys_module;
}

}// namespace py