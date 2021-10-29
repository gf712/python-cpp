#include "PyModule.hpp"


void PyModule::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_module_name);
	for (auto &[k, v] : m_module_definitions) {
		k->visit_graph(visitor);
		if (std::holds_alternative<PyObject *>(v)) {
			std::get<PyObject *>(v)->visit_graph(visitor);
		}
	}
}