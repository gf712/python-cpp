#pragma once

namespace py {
namespace sre {
	class Match : public PyBaseObject
	{
		friend class ::Heap;
		friend class py::detail::Allocator;

		Match();
		Match(PyType *);

	  public:
		static PyResult<Match *> create();

		void visit_graph(Visitor &visitor) override;

		static PyType *register_type(PyModule *module);
	};
}// namespace sre
}// namespace py