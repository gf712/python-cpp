#include "PyGenerator.hpp"
#include "MemoryError.hpp"
#include "PyCode.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyGenerator *as(PyObject *obj)
{
	if (obj->type() == generator()) { return static_cast<PyGenerator *>(obj); }
	return nullptr;
}

template<> const PyGenerator *as(const PyObject *obj)
{
	if (obj->type() == generator()) { return static_cast<const PyGenerator *>(obj); }
	return nullptr;
}

PyGenerator::PyGenerator(PyFrame *frame,
	std::unique_ptr<StackFrame> &&stack_frame,
	bool is_running,
	PyObject *code,
	PyString *name,
	PyString *qualname)
	: GeneratorInterface(BuiltinTypes::the().generator(),
		frame,
		std::move(stack_frame),
		is_running,
		code,
		name,
		qualname)
{}

PyResult<PyGenerator *> PyGenerator::create(PyFrame *frame,
	std::unique_ptr<StackFrame> &&stack_frame,
	PyString *name,
	PyString *qualname)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyGenerator>(
			frame, std::move(stack_frame), false, frame->code(), name, qualname)) {
		return Ok(obj);
	}
	return Err(memory_error(sizeof(PyGenerator)));
}

namespace {
	std::once_flag generator_flag;

	std::unique_ptr<TypePrototype> register_generator()
	{
		return std::move(klass<PyGenerator>(PyGenerator::GeneratorTypeName)
							 .def("close", &PyGenerator::close)
							 .def("send", &PyGenerator::send)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyGenerator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(generator_flag, []() { type = register_generator(); });
		return std::move(type);
	};
}

PyType *PyGenerator::type() const { return generator(); }

void PyGenerator::visit_graph(Visitor &visitor)
{
	GeneratorInterface<PyGenerator>::visit_graph(visitor);
}

}// namespace py
