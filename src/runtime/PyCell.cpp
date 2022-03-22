#include "PyCell.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyCell *as(PyObject *obj)
{
	if (obj->type() == cell()) { return static_cast<PyCell *>(obj); }
	return nullptr;
}

template<> const PyCell *as(const PyObject *obj)
{
	if (obj->type() == cell()) { return static_cast<const PyCell *>(obj); }
	return nullptr;
}

PyCell::PyCell(const Value &content) : PyBaseObject(BuiltinTypes::the().cell()), m_content(content)
{}

PyCell *PyCell::create() { return VirtualMachine::the().heap().allocate<PyCell>(nullptr); }

PyCell *PyCell::create(const Value &content)
{
	if (std::holds_alternative<PyObject *>(content)) { ASSERT(std::get<PyObject *>(content)) }
	return VirtualMachine::the().heap().allocate<PyCell>(content);
}

std::string PyCell::to_string() const
{
	if (std::holds_alternative<PyObject *>(m_content)) {
		if (auto *obj = std::get<PyObject *>(m_content)) {
			return fmt::format("<cell at {}: {} object at {}>",
				(void *)this,
				obj->type()->to_string(),
				(void *)obj);
		} else {
			return "<cell: empty>";
		}
	} else {
		return fmt::format("<cell at {}: {} object at {}>",
			(void *)this,
			PyObject::from(m_content)->type()->to_string(),
			(void *)&m_content);
	}
}

void PyCell::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (std::holds_alternative<PyObject *>(m_content)) {
		if (auto *obj = std::get<PyObject *>(m_content)) { visitor.visit(*obj); }
	}
}

PyType *PyCell::type() const { return cell(); }

const Value &PyCell::content() const { return m_content; }

PyObject *PyCell::__repr__() const { return PyString::from(String{ to_string() }); }

namespace {
	std::once_flag cell_flag;

	std::unique_ptr<TypePrototype> register_cell() { return std::move(klass<PyCell>("cell").type); }
}// namespace

std::unique_ptr<TypePrototype> PyCell::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(cell_flag, []() { type = register_cell(); });
	return std::move(type);
}

}// namespace py