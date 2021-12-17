#include "LoadAssertionError.hpp"
#include "runtime/AssertionError.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"

void LoadAssertionError::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.reg(m_assertion_location) = PyNativeFunction::create(
		"__assertion__", [&vm](PyTuple *args, PyDict *kwargs) -> PyObject * {
			ASSERT(args->size() < 2)
			// TODO: implement kwargs
			ASSERT(!kwargs)

			if (args->size() == 1) {
				const auto &value = args->elements()[0];
				ASSERT(std::holds_alternative<PyObject *>(value)
					   || std::holds_alternative<String>(value))
				if (std::holds_alternative<PyObject *>(value)) {
					auto *msg = std::get<PyObject *>(value)->__repr__();
					ASSERT(as<PyString>(msg))
					vm.interpreter().raise_exception(assertion_error(as<PyString>(msg)->value()));
				} else {
					vm.interpreter().raise_exception(assertion_error(std::get<String>(value).s));
				}
			} else {
				vm.interpreter().raise_exception(assertion_error(""));
			}
			return py_none();
		});
}