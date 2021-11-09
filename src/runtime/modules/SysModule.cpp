#include "runtime/PyList.hpp"
#include "runtime/PyModule.hpp"

#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

#include <filesystem>

static PyModule *s_sys_module = nullptr;

namespace {
PyList *create_sys_paths(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();

	auto *path_list = heap.allocate<PyList>();
	const auto &entry_script = interpreter.entry_script();
	path_list->append(PyString::create(std::filesystem::path(entry_script).parent_path()));

	return path_list;
}
}// namespace

PyModule *sys_module(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();

	if (s_sys_module && heap.slab().has_address(bit_cast<uint8_t *>(s_sys_module))) {
		return s_sys_module;
	}

	s_sys_module = heap.allocate<PyModule>(PyString::create("sys"));

	s_sys_module->insert(PyString::create("path"), create_sys_paths(interpreter));

	return s_sys_module;
}