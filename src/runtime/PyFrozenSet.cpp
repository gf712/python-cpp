#include "PyFrozenSet.hpp"
#include "MemoryError.hpp"
#include "PySet.hpp"
#include "StopIteration.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyFrozenSet *as(PyObject *obj)
{
	if (obj->type() == types::frozenset()) { return static_cast<PyFrozenSet *>(obj); }
	return nullptr;
}

template<> const PyFrozenSet *as(const PyObject *obj)
{
	if (obj->type() == types::frozenset()) { return static_cast<const PyFrozenSet *>(obj); }
	return nullptr;
}

PyFrozenSet::PyFrozenSet() : PyBaseObject(types::BuiltinTypes::the().frozenset()) {}

PyFrozenSet::PyFrozenSet(PyType *type) : PyBaseObject(type) {}

PyFrozenSet::PyFrozenSet(SetType elements) : PyFrozenSet() { m_elements = std::move(elements); }

PyResult<PyFrozenSet *> PyFrozenSet::create(SetType elements)
{
	auto *result = VirtualMachine::the().heap().allocate<PyFrozenSet>(elements);
	if (!result) { return Err(memory_error(sizeof(PyFrozenSet))); }
	return Ok(result);
}

namespace {
	PyFrozenSet *s_frozen_set = nullptr;
}

PyResult<PyFrozenSet *> PyFrozenSet::create()
{
	if (!s_frozen_set) {
		auto &heap = VirtualMachine::the().heap();
		s_frozen_set = heap.allocate_static<PyFrozenSet>().get();
		ASSERT(s_frozen_set);
	}
	return Ok(s_frozen_set);
}

std::string PyFrozenSet::to_string() const
{
	std::ostringstream os;

	os << "frozenset({";
	if (!m_elements.empty()) {
		auto it = m_elements.begin();
		while (std::next(it) != m_elements.end()) {
			std::visit(overloaded{
						   [&os](const auto &value) { os << value << ", "; },
						   [&os](PyObject *value) { os << value->to_string() << ", "; },
					   },
				*it);
			std::advance(it, 1);
		}
		std::visit(overloaded{
					   [&os](const auto &value) { os << value; },
					   [&os](PyObject *value) { os << value->to_string(); },
				   },
			*it);
	}
	os << "})";

	return os.str();
}

PyResult<PyObject *> PyFrozenSet::__new__(const PyType *type, PyTuple *args, PyDict *)
{
	ASSERT(type == types::frozenset());

	ASSERT(args);
	if (args->size() == 0) {
		return PyFrozenSet::create();
	} else if (args->size() == 1) {
		auto iterable_ = PyObject::from(args->elements()[0]);
		if (iterable_.is_err()) return iterable_;
		auto *iterable = iterable_.unwrap();

		auto iterator_ = iterable->iter();
		if (iterator_.is_err()) return iterator_;
		auto *iterator = iterator_.unwrap();

		auto value_ = iterator->next();

		SetType set;
		while (!value_.is_err()) {
			set.insert(value_.unwrap());
			value_ = iterator->next();
		}

		if (value_.unwrap_err()->type() != stop_iteration()->type()) { return value_; }

		return PyFrozenSet::create(set);
	} else {
		return Err(type_error("frozenset expected at most 1 argument, got {}", args->size()));
	}
}

PyResult<int32_t> PyFrozenSet::__init__(PyTuple *args, PyDict *kwargs)
{
	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("frozenset() takes no keyword arguments"));
	}
	if (!args || args->elements().size() == 0) { return Ok(0); }

	if (args->elements().size() != 1) {
		return Err(
			type_error("frozenset expected at most 1 argument, got {}", args->elements().size()));
	}

	auto iterable = PyObject::from(args->elements()[0]);
	if (iterable.is_err()) return Err(iterable.unwrap_err());

	auto iterator = iterable.unwrap()->iter();
	if (iterator.is_err()) return Err(iterator.unwrap_err());

	auto value = iterator.unwrap()->next();

	while (value.is_ok()) {
		m_elements.insert(value.unwrap());
		value = iterator.unwrap()->next();
	}

	if (value.unwrap_err()->type() != stop_iteration()->type()) { return Err(value.unwrap_err()); }

	return Ok(0);
}

PyResult<PyObject *> PyFrozenSet::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyFrozenSet::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	auto *it = heap.allocate<PySetIterator>(*this);
	if (!it) { return Err(memory_error(sizeof(PySetIterator))); }
	return Ok(it);
}

PyResult<size_t> PyFrozenSet::__len__() const { return Ok(m_elements.size()); }

PyResult<PyObject *> PyFrozenSet::__eq__(const PyObject *other) const
{
	(void)other;
	TODO();
	return Err(nullptr);
}

PyResult<bool> PyFrozenSet::__contains__(const PyObject *value) const
{
	const Value value_{ const_cast<PyObject *>(value) };
	return Ok(m_elements.contains(value_));
}


void PyFrozenSet::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto &el : m_elements) {
		if (std::holds_alternative<PyObject *>(el)) {
			if (std::get<PyObject *>(el) != this) visitor.visit(*std::get<PyObject *>(el));
		}
	}
}

PyType *PyFrozenSet::static_type() const { return types::frozenset(); }

namespace {

	std::once_flag frozenset_flag;

	std::unique_ptr<TypePrototype> register_frozenset()
	{
		return std::move(klass<PyFrozenSet>("frozenset").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyFrozenSet::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(frozenset_flag, []() { type = register_frozenset(); });
		return std::move(type);
	};
}
}// namespace py
