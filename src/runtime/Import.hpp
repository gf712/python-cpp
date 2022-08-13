#pragma once

#include "PyObject.hpp"

namespace py {

struct FrozenModule
	: NonCopyable
	, NonMoveable
{
	std::string_view name;
	const std::vector<uint8_t> &code;
	const bool is_package;
};

std::optional<std::reference_wrapper<const FrozenModule>> find_frozen(PyString *name);

PyResult<PyObject *> import_module_level_object(PyString *name,
	PyDict *globals,
	PyObject *locals,
	PyObject *fromlist,
	uint32_t level);

PyResult<PyModule *> import_frozen_module(PyString *name);

PyResult<PyObject *> create_builtin(PyObject *spec);
}// namespace py
