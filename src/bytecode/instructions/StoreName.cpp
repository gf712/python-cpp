#include "StoreName.hpp"

#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"


void StoreName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &value = vm.reg(m_source);
	auto *obj = std::visit(
		overloaded{ [](const Number &n) -> PyObject * { return PyNumber::create(n); },
			[](const String &s) -> PyObject * { return PyString::create(s.s); },
			[](const NameConstant &s) -> PyObject * { return PyObject::from(s); },
			[](PyObject *obj) -> PyObject * { return obj; },
			[&interpreter, this](const auto &) -> PyObject * {
				interpreter.raise_exception("Failed to store object \"{}\"", m_object_name);
				return nullptr;
			} },
		value);

	interpreter.store_object(m_object_name, obj);
}