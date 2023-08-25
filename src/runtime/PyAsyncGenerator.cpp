#include "PyAsyncGenerator.hpp"
#include "MemoryError.hpp"
#include "PyCode.hpp"
#include "PyString.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

template<> PyAsyncGenerator *as(PyObject *obj)
{
	if (obj->type() == types::async_generator()) { return static_cast<PyAsyncGenerator *>(obj); }
	return nullptr;
}

template<> const PyAsyncGenerator *as(const PyObject *obj)
{
	if (obj->type() == types::async_generator()) { return static_cast<const PyAsyncGenerator *>(obj); }
	return nullptr;
}

PyAsyncGenerator::PyAsyncGenerator(PyType *type) : GeneratorInterface(type) {}

PyAsyncGenerator::PyAsyncGenerator(PyFrame *frame,
	std::unique_ptr<StackFrame> &&stack_frame,
	bool is_running,
	PyObject *code,
	PyString *name,
	PyString *qualname)
	: GeneratorInterface(types::BuiltinTypes::the().async_generator(),
		frame,
		std::move(stack_frame),
		is_running,
		code,
		name,
		qualname)
{}

PyResult<PyAsyncGenerator *> PyAsyncGenerator::create(PyFrame *frame,
	std::unique_ptr<StackFrame> &&stack_frame,
	PyString *name,
	PyString *qualname)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyAsyncGenerator>(
			frame, std::move(stack_frame), false, frame->code(), name, qualname)) {
		return Ok(obj);
	}
	return Err(memory_error(sizeof(PyAsyncGenerator)));
}

namespace {
	std::once_flag async_generator_flag;

	std::unique_ptr<TypePrototype> register_async_generator()
	{
		return std::move(klass<PyAsyncGenerator>(PyAsyncGenerator::GeneratorTypeName)
							 .def("close", &PyAsyncGenerator::close)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyAsyncGenerator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(async_generator_flag, []() { type = register_async_generator(); });
		return std::move(type);
	};
}

PyType *PyAsyncGenerator::static_type() const { return types::async_generator(); }

void PyAsyncGenerator::visit_graph(Visitor &visitor)
{
	GeneratorInterface<PyAsyncGenerator>::visit_graph(visitor);
}

}// namespace py
