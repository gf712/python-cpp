#pragma once

#include "GeneratorInterface.hpp"

namespace py {
class PyCoroutine : public GeneratorInterface<PyCoroutine>
{
	friend ::Heap;

	PyCoroutine(PyFrame *m_frame,
		std::unique_ptr<StackFrame> &&,
		bool is_running,
		PyObject *m_code,
		PyString *m_name,
		PyString *m_qualname);

  public:
	static constexpr std::string_view GeneratorTypeName = "coroutine";

  public:
	static PyResult<PyCoroutine *>
		create(PyFrame *frame, std::unique_ptr<StackFrame> &&, PyString *name, PyString *qualname);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
	void visit_graph(Visitor &visitor) override;
};
}// namespace py