module;
#include "memory/allocate.hpp"

export module py.runtime:none;
import :object;
import std;
import py.memory;

export namespace py {
class PyType;

class PyNone : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend PyObject *py_none();

	bool m_value;

	PyNone(PyType *);

  public:
	std::string to_string() const override;

	PyResult<PyObject *> __add__(const PyObject *obj) const;
	PyResult<PyObject *> __repr__() const;
	PyResult<bool> __bool__() const;

	PyResult<bool> true_() final;

	bool value() const { return m_value; }

	void visit_graph(Visitor &) override {}

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	static PyNone *create();
	PyNone();
};

PyObject *py_none();

}// namespace py
