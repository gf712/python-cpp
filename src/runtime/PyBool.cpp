#include "PyBool.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyBool *as(PyObject *node)
{
	if (node->type() == bool_()) { return static_cast<PyBool *>(node); }
	return nullptr;
}

template<> const PyBool *as(const PyObject *node)
{
	if (node->type() == bool_()) { return static_cast<const PyBool *>(node); }
	return nullptr;
}

PyBool::PyBool(bool value) : PyInteger(BuiltinTypes::the().bool_(), value) {}

std::string PyBool::to_string() const
{
	ASSERT(std::holds_alternative<int64_t>(m_value.value));
	return std::get<int64_t>(m_value.value) ? "True" : "False";
}

bool PyBool::value() const
{
	ASSERT(std::holds_alternative<int64_t>(m_value.value));
	return std::get<int64_t>(m_value.value);
}

PyResult<PyObject *> PyBool::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().size() == 0)
	ASSERT(args && args->size() == 1)
	ASSERT(type == py::bool_())

	const auto &value = PyObject::from(args->elements()[0]);

	if (value.is_err()) return value;

	if (value.unwrap()->type() == py::bool_()) return value;

	return value.unwrap()->bool_().and_then(
		[](const auto &v) { return Ok(v ? py_true() : py_false()); });
}

PyResult<PyObject *> PyBool::__repr__() const { return PyString::create(to_string()); }

PyResult<bool> PyBool::__bool__() const
{
	ASSERT(std::holds_alternative<int64_t>(m_value.value));
	return Ok(static_cast<bool>(std::get<int64_t>(m_value.value)));
}

PyResult<PyBool *> PyBool::create(bool value)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate_static<PyBool>(value).get();
	ASSERT(result)
	return Ok(result);
}

PyType *PyBool::type() const { return py::bool_(); }

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
		return std::move(klass<PyBool>("bool", integer()).type);
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
