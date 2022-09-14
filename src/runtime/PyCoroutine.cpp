#include "PyCoroutine.hpp"
#include "MemoryError.hpp"
#include "PyCode.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> PyCoroutine *as(PyObject *obj)
{
	if (obj->type() == coroutine()) { return static_cast<PyCoroutine *>(obj); }
	return nullptr;
}

template<> const PyCoroutine *as(const PyObject *obj)
{
	if (obj->type() == coroutine()) { return static_cast<const PyCoroutine *>(obj); }
	return nullptr;
}

PyCoroutine::PyCoroutine(PyFrame *frame,
	std::unique_ptr<StackFrame> &&stack_frame,
	bool is_running,
	PyObject *code,
	PyString *name,
	PyString *qualname)
	: GeneratorInterface(BuiltinTypes::the().coroutine(),
		frame,
		std::move(stack_frame),
		is_running,
		code,
		name,
		qualname)
{}

PyResult<PyCoroutine *> PyCoroutine::create(PyFrame *frame,
	std::unique_ptr<StackFrame> &&stack_frame,
	PyString *name,
	PyString *qualname)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyCoroutine>(
			frame, std::move(stack_frame), false, frame->code(), name, qualname)) {
		return Ok(obj);
	}
	return Err(memory_error(sizeof(PyCoroutine)));
}

namespace {
	std::once_flag coroutine_flag;

	std::unique_ptr<TypePrototype> register_coroutine()
	{
		return std::move(klass<PyCoroutine>(PyCoroutine::GeneratorTypeName)
							 .def("close", &PyCoroutine::close)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyCoroutine::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(coroutine_flag, []() { type = register_coroutine(); });
		return std::move(type);
	};
}

PyType *PyCoroutine::type() const { return coroutine(); }

void PyCoroutine::visit_graph(Visitor &visitor)
{
	GeneratorInterface<PyCoroutine>::visit_graph(visitor);
}

}// namespace py