module;
#include "memory/allocate.hpp"

export module py.runtime:not_implemented;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyType;

class NotImplemented : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend NotImplemented *not_implemented();

	NotImplemented();
	NotImplemented(PyType *type);

	static PyResult<NotImplemented *> create();

  public:
	std::string to_string() const override;

	void visit_graph(Visitor &) override {}

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

NotImplemented *not_implemented();

}// namespace py
