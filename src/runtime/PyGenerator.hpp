#pragma once

#include "PyFrame.hpp"
#include "PyObject.hpp"

namespace py {
class PyGenerator : public PyBaseObject
{
	friend ::Heap;

	PyFrame *m_frame{ nullptr };
	std::unique_ptr<StackFrame> m_stack_frame;
	bool m_is_running{ false };
	PyObject *m_code{ nullptr };
	PyString *m_name{ nullptr };
	PyString *m_qualname{ nullptr };
	std::unique_ptr<std::vector<PyFrame::ExceptionStackItem>> m_exception_stack;
	bool m_invalid_return{ false };

	PyGenerator(PyFrame *m_frame,
		std::unique_ptr<StackFrame> &&,
		bool is_running,
		PyObject *m_code,
		PyString *m_name,
		PyString *m_qualname);

  public:
	static PyResult<PyGenerator *>
		create(PyFrame *frame, std::unique_ptr<StackFrame> &&, PyString *name, PyString *qualname);

	void set_invalid_return(bool invalid_return) { m_invalid_return = invalid_return; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();
	PyResult<PyObject *> send();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
	void visit_graph(Visitor &visitor) override;
};
}// namespace py