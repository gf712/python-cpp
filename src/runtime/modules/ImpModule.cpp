#include "Modules.hpp"
#include "config.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/Import.hpp"
#include "runtime/ImportError.hpp"
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
				ASSERT(args->size() == 1)
				ASSERT(!kwargs || kwargs->map().size() == 0)

				auto arg0 = PyObject::from(args->elements()[0]);
				if (arg0.is_err()) { return Err(arg0.unwrap_err()); }

				if (!as<PyString>(arg0.unwrap())) {
					return Err(type_error("expected name to be a string, but got {}",
						arg0.unwrap()->type()->to_string()));
				}

				return Ok(
					find_frozen(as<PyString>(arg0.unwrap())).has_value() ? py_true() : py_false());
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("is_frozen_package").unwrap(),
		PyNativeFunction::create("is_frozen_package",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(args->size() == 1)
				ASSERT(!kwargs || kwargs->map().size() == 0)

				auto arg0 = PyObject::from(args->elements()[0]);
				if (arg0.is_err()) { return Err(arg0.unwrap_err()); }

				if (!as<PyString>(arg0.unwrap())) {
					return Err(type_error("expected name to be a string, but got {}",
						arg0.unwrap()->type()->to_string()));
				}

				if (auto frozen_module = find_frozen(as<PyString>(arg0.unwrap()))) {
					return Ok(frozen_module->get().is_package ? py_true() : py_false());
				} else {
					return Err(import_error(
						"No such frozen object named {}", as<PyString>(arg0.unwrap())->value()));
				}
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("get_frozen_object").unwrap(),
		PyNativeFunction::create("get_frozen_object",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(args->size() == 1)
				ASSERT(!kwargs || kwargs->map().size() == 0)

				auto arg0 = PyObject::from(args->elements()[0]);
				if (arg0.is_err()) { return Err(arg0.unwrap_err()); }

				if (!as<PyString>(arg0.unwrap())) {
					return Err(type_error("expected name to be a string, but got {}",
						arg0.unwrap()->type()->to_string()));
				}

				if (auto frozen_module = find_frozen(as<PyString>(arg0.unwrap()))) {
					std::shared_ptr<Program> program =
						BytecodeProgram::deserialize(frozen_module->get().code);
					return PyCode::create(program);
				} else {
					return Err(import_error(
						"No such frozen object named {}", as<PyString>(arg0.unwrap())->value()));
				}
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("acquire_lock").unwrap(),
		PyNativeFunction::create("acquire_lock",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->size() == 0)
				ASSERT(!kwargs || kwargs->map().size() == 0)
				return Ok(py_none());
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("release_lock").unwrap(),
		PyNativeFunction::create("release_lock",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->size() == 0)
				ASSERT(!kwargs || kwargs->map().size() == 0)
				return Ok(py_none());
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("is_builtin").unwrap(),
		PyNativeFunction::create("is_builtin",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->size() == 1)
				ASSERT(!kwargs || kwargs->map().size() == 0)
				auto name = PyObject::from(args->elements()[0]).unwrap();
				ASSERT(as<PyString>(name))

				return is_builtin(as<PyString>(name)->value()) ? Ok(py_true()) : Ok(py_false());
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("create_builtin").unwrap(),
		PyNativeFunction::create("create_builtin",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->size() == 1)
				ASSERT(!kwargs || kwargs->map().size() == 0)
				auto spec = PyObject::from(args->elements()[0]).unwrap();

				return create_builtin(spec);
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("exec_builtin").unwrap(),
		PyNativeFunction::create("exec_builtin",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->size() == 1)
				ASSERT(!kwargs || kwargs->map().size() == 0)
				auto mod = PyObject::from(args->elements()[0]).unwrap();
				return exec_builtin(mod);
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("exec_dynamic").unwrap(),
		PyNativeFunction::create("exec_dynamic",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->size() == 1)
				ASSERT(!kwargs || kwargs->map().size() == 0)
				auto mod = PyObject::from(args->elements()[0]).unwrap();
				return exec_builtin(mod);
			})
			.unwrap());

	s_imp_module->add_symbol(PyString::create("extension_suffixes").unwrap(),
		PyNativeFunction::create("extension_suffixes",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!args || args->elements().empty())
				ASSERT(!kwargs || kwargs->map().size() == 0)
				return PyList::create();
			})
			.unwrap());

	return s_imp_module;
}
}// namespace py