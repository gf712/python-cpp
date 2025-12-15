#include "PyEllipsis.hpp"
#include "MemoryError.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {
PyEllipsis::PyEllipsis(PyType *type) : PyBaseObject(type) {}

PyEllipsis::PyEllipsis() : PyBaseObject(types::BuiltinTypes::the().ellipsis()) {}

PyResult<PyEllipsis *> PyEllipsis::create()
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate_static<PyEllipsis>();
	if (!obj) { return Err(memory_error(sizeof(PyEllipsis))); }
	return Ok(obj);
}

PyResult<PyObject *> PyEllipsis::__add__(const PyObject *) const { TODO(); }

PyResult<PyObject *> PyEllipsis::__repr__() const { return PyString::create("Ellipsis"); }

PyType *PyEllipsis::static_type() const { return types::ellipsis(); }

PyObject *py_ellipsis()
{
	static PyObject *ellipsis = nullptr;
	if (!ellipsis) {
		auto obj = PyEllipsis::create();
		ASSERT(obj.is_ok());
		ellipsis = obj.unwrap();
	}
	return ellipsis;
}

namespace {

	std::once_flag ellipsis_flag;

	std::unique_ptr<TypePrototype> register_ellipsis()
	{
		return std::move(klass<PyEllipsis>("ellipsis").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyEllipsis::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(ellipsis_flag, []() { type = register_ellipsis(); });
		return std::move(type);
	};
}
}// namespace py
