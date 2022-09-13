#pragma once

#include "PyFrame.hpp"
#include "PyObject.hpp"

namespace py {

template<typename GeneratorType> class GeneratorInterface : public PyBaseObject
{
	PyFrame *m_frame{ nullptr };
	std::unique_ptr<StackFrame> m_stack_frame;
	bool m_is_running{ false };
	PyObject *m_code{ nullptr };
	PyString *m_name{ nullptr };
	PyString *m_qualname{ nullptr };
	std::unique_ptr<std::vector<PyFrame::ExceptionStackItem>> m_exception_stack;
	bool m_invalid_return{ false };

  protected:
	GeneratorInterface(TypePrototype &type,
		PyFrame *frame,
		std::unique_ptr<StackFrame> &&,
		bool is_running,
		PyObject *code,
		PyString *name,
		PyString *qualname);

  public:
	void set_invalid_return(bool invalid_return) { m_invalid_return = invalid_return; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();
	PyResult<PyObject *> send();

	void visit_graph(Visitor &visitor) override;
};

}// namespace py