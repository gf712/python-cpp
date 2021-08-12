#include "StoreName.hpp"

#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"


void StoreName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &value = vm.reg(m_source);
	auto obj = std::visit(
		overloaded{
			[](const Number &n) { return std::static_pointer_cast<PyObject>(PyNumber::create(n)); },
			[](const String &s) {
				return std::static_pointer_cast<PyObject>(PyString::create(s.s));
			},
			[](const NameConstant &s) { return PyObject::from(s); },
			[](const std::shared_ptr<PyObject> &obj) { return obj; },
			[&interpreter, this](const auto &) {
				interpreter.raise_exception("Failed to store object \"{}\"", m_object_name);
				return std::shared_ptr<PyObject>(nullptr);
			} },
		value);

	interpreter.store_object(m_object_name, obj);
}