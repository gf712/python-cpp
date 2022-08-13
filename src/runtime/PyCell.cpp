#include "PyCell.hpp"
#include "MemoryError.hpp"
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

PyResult<PyCell *> PyCell::create()
{
	auto *obj = VirtualMachine::the().heap().allocate<PyCell>(nullptr);
	if (!obj) { return Err(memory_error(sizeof(PyCell))); }
	return Ok(obj);
}

PyResult<PyCell *> PyCell::create(const Value &content)
{
	if (std::holds_alternative<PyObject *>(content)) { ASSERT(std::get<PyObject *>(content)) }
	auto *obj = VirtualMachine::the().heap().allocate<PyCell>(content);
	if (!obj) { return Err(memory_error(sizeof(PyCell))); }
	return Ok(obj);
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
			PyObject::from(m_content).unwrap()->type()->to_string(),
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

PyResult<PyObject *> PyCell::__repr__() const { return PyString::create(to_string()); }

namespace {
	std::once_flag cell_flag;

	std::unique_ptr<TypePrototype> register_cell() { return std::move(klass<PyCell>("cell").type); }
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyCell::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(cell_flag, []() { type = register_cell(); });
		return std::move(type);
	};
}

}// namespace py