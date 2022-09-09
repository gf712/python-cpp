#pragma once

#include "PyObject.hpp"

namespace py {
class PyGenerator : public PyBaseObject
{
	friend ::Heap;

	PyFrame *m_frame{ nullptr };
	bool m_is_running{ false };
	PyObject *m_code{ nullptr };
	PyString *m_name{ nullptr };
	PyString *m_qualname{ nullptr };

	PyGenerator(PyFrame *m_frame,
		bool is_running,
		PyObject *m_code,
		PyString *m_name,
		PyString *m_qualname);

  public:
	static PyResult<PyGenerator *> create(PyFrame *frame, PyString *name, PyString *qualname);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;
	void visit_graph(Visitor &visitor) override;
};
}// namespace py