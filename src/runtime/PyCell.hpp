#pragma once

#include "PyObject.hpp"

namespace py {

class PyCell : public PyBaseObject
{
	friend class ::Heap;

	Value m_content;

  protected:
	PyCell(const Value &);

  public:
	static PyResult<PyCell *> create();
	static PyResult<PyCell *> create(const Value &);

	std::string to_string() const override;
	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;

	void set_cell(const Value&);

	const Value &content() const;

	PyResult<PyObject *> __repr__() const;
};

}// namespace py