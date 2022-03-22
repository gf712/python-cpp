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
	static PyCell *create();
	static PyCell *create(const Value &);

	std::string to_string() const override;
	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

	const Value &content() const;

	PyObject *__repr__() const;
};

}// namespace py