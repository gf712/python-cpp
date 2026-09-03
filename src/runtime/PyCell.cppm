module;

#include "memory/allocate.hpp"

export module py.runtime:cell;

import :object;
import std;
import py.memory;

export namespace py {
class PyType;

class PyCell : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	Value m_content{ nullptr };

  protected:
	PyCell(PyType *);

	PyCell(const Value &);

  public:
	static PyResult<PyCell *> create();
	static PyResult<PyCell *> create(const Value &);

	std::string to_string() const override;
	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	void set_cell(const Value &);

	const Value &content() const;

	bool empty() const;

	PyResult<PyObject *> __repr__() const;
};

}// namespace py
