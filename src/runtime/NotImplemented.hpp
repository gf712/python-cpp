#pragma once

#include "PyObject.hpp"

namespace py {

class NotImplemented : public PyBaseObject
{
  public:
	std::string to_string() const override;

	void visit_graph(Visitor &) override {}

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;

  private:
	NotImplemented() = delete;
};

}// namespace py