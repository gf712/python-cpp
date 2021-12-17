#include "PySlotWrapper.hpp"
#include "PyFunction.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

std::string PySlotWrapper::to_string() const
{
	return fmt::format(
		"<slot wrapper '{}' of '{}' objects>", m_name->to_string(), m_slot_type->name());
}

void PySlotWrapper::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	visitor.visit(*m_slot_type);
}

PyObject *PySlotWrapper::__repr__() const { return PyString::create(to_string()); }

PyObject *PySlotWrapper::__call__(PyTuple *args, PyDict *kwargs)
{
	// split args tuple -> (args[0], args[1:])
	// since args[0] is self (right?)
	std::vector<Value> new_args_vector;
	new_args_vector.reserve(args->size() - 1);
	PyObject *self = PyObject::from(args->elements()[0]);
	for (size_t i = 1; i < args->size(); ++i) { new_args_vector.push_back(args->elements()[i]); }
	args = PyTuple::create(new_args_vector);
	auto *result = m_slot(self, args, kwargs);

	// FIXME: this should be independent from the VM's registers
	VirtualMachine::the().reg(0) = result;

	return result;
}

PyObject *PySlotWrapper::__get__(PyObject *instance, PyObject * /*owner*/) const
{
	if (!instance) { return const_cast<PySlotWrapper *>(this); }
	if (instance->type() != m_slot_type) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("descriptor '{}' for '{}' objects "
					   "doesn't apply to a '{}' object",
				m_name->value(),
				m_slot_type->underlying_type().__name__,
				instance->type()->underlying_type().__name__));
		return nullptr;
	}
	return PyNativeFunction::create(
		m_name->value(),
		[this](PyTuple *args, PyDict *kwargs) {
			return const_cast<PySlotWrapper *>(this)->__call__(args, kwargs);
		},
		const_cast<PySlotWrapper *>(this),
		instance);
}

PySlotWrapper::PySlotWrapper(PyString *name,
	PyType *slot_type,
	std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function)
	: PyBaseObject(BuiltinTypes::the().slot_wrapper()), m_name(std::move(name)),
	  m_slot_type(slot_type), m_slot(std::move(function))
{}

PySlotWrapper *PySlotWrapper::create(PyString *name,
	PyType *slot_type,
	std::function<PyObject *(PyObject *, PyTuple *, PyDict *)> function)
{
	ASSERT(name)
	ASSERT(slot_type)
	return VirtualMachine::the().heap().allocate<PySlotWrapper>(name, slot_type, function);
}

PyType *PySlotWrapper::type() const { return slot_wrapper(); }

namespace {

std::once_flag slot_wrapper_flag;

std::unique_ptr<TypePrototype> register_slot_wrapper()
{
	return std::move(klass<PySlotWrapper>("slot_wrapper").type);
}
}// namespace

std::unique_ptr<TypePrototype> PySlotWrapper::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(slot_wrapper_flag, []() { type = ::register_slot_wrapper(); });
	return std::move(type);
}

template<> PySlotWrapper *as(PyObject *obj)
{
	if (obj->type() == slot_wrapper()) { return static_cast<PySlotWrapper *>(obj); }
	return nullptr;
}

template<> const PySlotWrapper *as(const PyObject *obj)
{
	if (obj->type() == slot_wrapper()) { return static_cast<const PySlotWrapper *>(obj); }
	return nullptr;
}