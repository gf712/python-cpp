#include "PyIterator.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/Value.hpp"
#include "runtime/ValueError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include <limits>


using namespace py;

PyIterator::PyIterator(PyObject *iterator)
	: PyBaseObject(types::BuiltinTypes::the().iterator()), m_iterator(iterator)
{}

PyResult<PyIterator *> PyIterator::create(PyObject *iterator)
{
	auto *result = VirtualMachine::the().heap().allocate<PyIterator>(iterator);
	if (!result) { return Err(memory_error(sizeof(PyIterator))); }
	return Ok(result);
}

PyResult<PyObject *> PyIterator::__iter__() const { return Ok(const_cast<PyIterator *>(this)); }

PyResult<PyObject *> PyIterator::__next__()
{
	if (!m_iterator) { return Ok(py_none()); }

	if (m_index == std::numeric_limits<size_t>::max()) {
		return Err(value_error("iter index too large"));
	}

	auto result = m_iterator->getitem(m_index);
	if (result.is_err()
		&& (result.unwrap_err()->type()->issubclass(types::index_error())
			|| result.unwrap_err()->type()->issubclass(types::stop_iteration()))) {
		m_iterator = nullptr;
		return Err(stop_iteration());
	} else if (result.is_err()) {
		return result;
	}

	m_index++;
	return result;
}

PyResult<size_t> PyIterator::__len__() const
{
	if (!m_iterator) { return Ok(size_t{ 0 }); }

	if (m_iterator->type()->underlying_type().mapping_type_protocol.has_value()
		&& m_iterator->type()->underlying_type().mapping_type_protocol->__len__.has_value()) {
		return m_iterator->as_mapping()
			.and_then([](auto mapping) { return mapping.len(); })
			.and_then([this](size_t seqsize) -> PyResult<size_t> { return Ok(seqsize - m_index); });
	}

	return Ok(0);
}

void PyIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_iterator) { visitor.visit(*m_iterator); }
}

PyType *PyIterator::static_type() const { return types::iterator(); }


namespace {
std::once_flag iterator_flag;

std::unique_ptr<TypePrototype> register_iterator()
{
	return std::move(klass<PyIterator>("iterator").disable_new().type);
}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(iterator_flag, []() { type = register_iterator(); });
		return std::move(type);
	};
}