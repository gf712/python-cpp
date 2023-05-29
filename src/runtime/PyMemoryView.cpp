#include "PyMemoryView.hpp"
#include "runtime/PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

PyMemoryView::PyMemoryView(PyType *type) : PyBaseObject(type) {}

PyMemoryView::PyMemoryView(PyType *type, PyObject *object) : PyBaseObject(type), m_object(object) {}

PyResult<PyObject *> PyMemoryView::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
		kwargs,
		"memoryview",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 1>{});

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [object] = result.unwrap();

	auto obj =
		VirtualMachine::the().heap().allocate<PyMemoryView>(const_cast<PyType *>(type), object);
	if (!obj) { return Err(memory_error(sizeof(PyMemoryView))); }
	return Ok(obj);
}

PyResult<PyObject *> PyMemoryView::__repr__() const { return PyString::create(to_string()); }

namespace {
	std::once_flag memoryview_flag;

	std::unique_ptr<TypePrototype> register_memoryview()
	{
		return std::move(klass<PyMemoryView>("memoryview").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyMemoryView::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(memoryview_flag, []() { type = register_memoryview(); });
		return std::move(type);
	};
}

PyType *PyMemoryView::static_type() const { return types::memoryview(); }

void PyMemoryView::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_object) { visitor.visit(*m_object); }
}

std::string PyMemoryView::to_string() const
{
	return fmt::format("<memory at {}>", static_cast<const void *>(this));
}

}// namespace py
