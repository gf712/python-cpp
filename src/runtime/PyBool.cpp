#include "PyBool.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyBool *as(PyObject *node)
{
	if (node->type() == types::bool_()) { return static_cast<PyBool *>(node); }
	return nullptr;
}

template<> const PyBool *as(const PyObject *node)
{
	if (node->type() == types::bool_()) { return static_cast<const PyBool *>(node); }
	return nullptr;
}

PyBool::PyBool(PyType *type) : PyInteger(type) {}

PyBool::PyBool(bool value) : PyInteger(types::BuiltinTypes::the().bool_(), value) {}

std::string PyBool::to_string() const { return value() ? "True" : "False"; }

bool PyBool::value() const
{
	ASSERT(std::holds_alternative<BigIntType>(m_value.value));
	return static_cast<bool>(std::get<BigIntType>(m_value.value));
}

PyResult<PyObject *> PyBool::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().size() == 0)
	ASSERT(args && args->size() == 1)
	ASSERT(type == types::bool_())

	const auto &value = PyObject::from(args->elements()[0]);

	if (value.is_err()) return value;

	if (value.unwrap()->type() == types::bool_()) return value;

	return value.unwrap()->true_().and_then(
		[](const auto &v) { return Ok(v ? py_true() : py_false()); });
}

PyResult<PyObject *> PyBool::__repr__() const { return PyString::create(to_string()); }

PyResult<bool> PyBool::true_()
{
	ASSERT(std::holds_alternative<BigIntType>(m_value.value));
	return Ok(value());
}

PyResult<PyBool *> PyBool::create(bool value)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate_static<PyBool>(value);
	ASSERT(result);
	return Ok(result);
}

PyType *PyBool::static_type() const { return types::bool_(); }

PyObject *py_true()
{
	static PyObject *value = nullptr;

	if (!value) { value = PyBool::create(true).unwrap(); }

	return value;
}

PyObject *py_false()
{
	static PyObject *value = nullptr;

	if (!value) { value = PyBool::create(false).unwrap(); }

	return value;
}

namespace {

	std::once_flag bool_flag;

	std::unique_ptr<TypePrototype> register_bool()
	{
		return std::move(klass<PyBool>("bool", types::integer()).type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyBool::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(bool_flag, []() { type = register_bool(); });
		return std::move(type);
	};
}

static_assert(std::is_same_v<PyBool::_InterfaceT, PyNumber>);
static_assert(std::is_same_v<PyBool::_InterfacingT, PyInteger>);
static_assert(!concepts::HasInterface<PyBool>);
}// namespace py
