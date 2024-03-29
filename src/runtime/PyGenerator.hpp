#pragma once

#include "GeneratorInterface.hpp"

namespace py {
class PyGenerator : public GeneratorInterface<PyGenerator>
{
	friend ::Heap;

	PyGenerator(PyType *);

	PyGenerator(PyFrame *m_frame,
		std::unique_ptr<StackFrame> &&,
		bool is_running,
		PyObject *m_code,
		PyString *m_name,
		PyString *m_qualname);

  public:
	static constexpr std::string_view GeneratorTypeName = "generator";

  public:
	static PyResult<PyGenerator *>
		create(PyFrame *frame, std::unique_ptr<StackFrame> &&, PyString *name, PyString *qualname);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
	void visit_graph(Visitor &visitor) override;
};
}// namespace py
