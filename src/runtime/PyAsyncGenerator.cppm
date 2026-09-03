module;

#include "memory/allocate.hpp"

export module py.runtime:asyncgenerator;
import :object;
import :detail;
import py.memory;
import std;

export struct StackFrame;

export namespace py {
class PyType;

class PyAsyncGenerator : public GeneratorInterface<PyAsyncGenerator>
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	PyAsyncGenerator(PyType *);

	PyAsyncGenerator(PyFrame *m_frame,
		std::unique_ptr<StackFrame> &&,
		bool is_running,
		PyObject *m_code,
		PyString *m_name,
		PyString *m_qualname);

  public:
	static constexpr std::string_view GeneratorTypeName = "async_generator";

  public:
	static PyResult<PyAsyncGenerator *>
		create(PyFrame *frame, std::unique_ptr<StackFrame> &&, PyString *name, PyString *qualname);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
	void visit_graph(Visitor &visitor) override;
};
}// namespace py
