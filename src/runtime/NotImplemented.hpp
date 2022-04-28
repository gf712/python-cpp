#pragma once

#include "PyObject.hpp"

namespace py {

class NotImplemented : public PyBaseObject
{
  public:
	std::string to_string() const override;

	void visit_graph(Visitor &) override {}

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

  private:
	NotImplemented() = delete;
};

}// namespace py