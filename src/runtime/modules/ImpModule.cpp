#include "Modules.hpp"
#include "config.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/Import.hpp"
#include "runtime/ImportError.hpp"
#include "runtime/PyArgParser.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyType.hpp"
#include "runtime/TypeError.hpp"
#include "vm/VM.hpp"

namespace py {
namespace {
	PyResult<PyObject *> exec_builtin(PyObject *)
	{
		// TODO: check implementation exec_builtin_or_dynamic in cpython's import.c
		return PyInteger::create(0);
	}
}// namespace

PyModule *imp_module()
{
	auto *s_imp_module = PyModule::create(
		PyDict::create().unwrap(), PyString::create("_imp").unwrap(), PyString::create("").unwrap())
							 .unwrap();

	s_imp_module->add_symbol(PyString::create("is_frozen").unwrap(),
		PyNativeFunction::create("is_frozen",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
					kwargs,
					"is_frozen",
					std::integral_constant<size_t, 1>{},
					std::integral_constant<size_t, 1>{});
				if (result.is_err()) return Err(result.unwrap_err());
				auto *name = std::get<0>(result.unwrap());

				if (!as<PyString>(name)) {
					return Err(type_error(
						"expected name to be a string, but got {}", name->type()->to_string()));
				}

				return Ok(find_frozen(as<PyString>(name)).has_value() ? py_true() : py_false());
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("is_frozen_package").unwrap(),
		PyNativeFunction::create("is_frozen_package",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
					kwargs,
					"is_frozen_package",
					std::integral_constant<size_t, 1>{},
					std::integral_constant<size_t, 1>{});
				if (result.is_err()) return Err(result.unwrap_err());
				auto *name = std::get<0>(result.unwrap());

				if (!as<PyString>(name)) {
					return Err(type_error(
						"expected name to be a string, but got {}", name->type()->to_string()));
				}

				if (auto frozen_module = find_frozen(as<PyString>(name))) {
					return Ok(frozen_module->get().is_package ? py_true() : py_false());
				} else {
					return Err(import_error(
						"No such frozen object named {}", as<PyString>(name)->value()));
				}
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("get_frozen_object").unwrap(),
		PyNativeFunction::create("get_frozen_object",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
					kwargs,
					"get_frozen_object",
					std::integral_constant<size_t, 1>{},
					std::integral_constant<size_t, 1>{});
				if (result.is_err()) return Err(result.unwrap_err());
				auto *name = std::get<0>(result.unwrap());

				if (!as<PyString>(name)) {
					return Err(type_error(
						"expected name to be a string, but got {}", name->type()->to_string()));
				}

				if (auto frozen_module = find_frozen(as<PyString>(name))) {
					std::shared_ptr<Program> program =
						BytecodeProgram::deserialize(frozen_module->get().code);
					return PyCode::create(program);
				} else {
					return Err(import_error(
						"No such frozen object named {}", as<PyString>(name)->value()));
				}
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("acquire_lock").unwrap(),
		PyNativeFunction::create("acquire_lock",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->size() == 0);
				ASSERT(!kwargs || kwargs->map().size() == 0);
				return Ok(py_none());
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("release_lock").unwrap(),
		PyNativeFunction::create("release_lock",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->size() == 0);
				ASSERT(!kwargs || kwargs->map().size() == 0);
				return Ok(py_none());
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("is_builtin").unwrap(),
		PyNativeFunction::create("is_builtin",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				auto result = PyArgsParser<PyString *>::unpack_tuple(args,
					kwargs,
					"is_builtin",
					std::integral_constant<size_t, 1>{},
					std::integral_constant<size_t, 1>{});
				if (result.is_err()) return Err(result.unwrap_err());
				auto *name = std::get<0>(result.unwrap());

				return is_builtin(name->value()) ? Ok(py_true()) : Ok(py_false());
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("create_builtin").unwrap(),
		PyNativeFunction::create("create_builtin",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
					kwargs,
					"create_builtin",
					std::integral_constant<size_t, 1>{},
					std::integral_constant<size_t, 1>{});
				if (result.is_err()) return Err(result.unwrap_err());

				return create_builtin(std::get<0>(result.unwrap()));
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("exec_builtin").unwrap(),
		PyNativeFunction::create("exec_builtin",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
					kwargs,
					"exec_builtin",
					std::integral_constant<size_t, 1>{},
					std::integral_constant<size_t, 1>{});
				if (result.is_err()) return Err(result.unwrap_err());
				return exec_builtin(std::get<0>(result.unwrap()));
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("exec_dynamic").unwrap(),
		PyNativeFunction::create("exec_dynamic",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
					kwargs,
					"exec_dynamic",
					std::integral_constant<size_t, 1>{},
					std::integral_constant<size_t, 1>{});
				if (result.is_err()) return Err(result.unwrap_err());
				return exec_builtin(std::get<0>(result.unwrap()));
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("extension_suffixes").unwrap(),
		PyNativeFunction::create("extension_suffixes",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->elements().empty());
				ASSERT(!kwargs || kwargs->map().size() == 0);
				return PyList::create();
			})
			.unwrap());

	return s_imp_module;
}
}// namespace py