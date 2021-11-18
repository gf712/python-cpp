#include "PyType.hpp"
#include "PyFunction.hpp"

void PyBoundMethod::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_self);
	visitor.visit(*m_method);
}

std::string PyBoundMethod::to_string() const
{
	return fmt::format("<bound method '{}' of {}>",
		m_method->name(),
		m_self->attributes().at("__qualname__")->to_string());
}


void PyMethodDescriptor::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	visitor.visit(*m_underlying_type);
	for (auto *capture : m_captures) { visitor.visit(*capture); }
}

std::string PyMethodDescriptor::to_string() const
{
	return fmt::format(
		"<method '{}' of '{}' objects>", m_name->to_string(), m_underlying_type->name()->value());
}

std::string PySlotWrapper::to_string() const
{
	return fmt::format("<slot wrapper '{}' of '{}' objects>",
		m_name->to_string(),
		m_underlying_type->name()->value());
}

void PySlotWrapper::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(*m_name);
	visitor.visit(*m_underlying_type);
}