#include "PySet.hpp"
#include "KeyError.hpp"
#include "MemoryError.hpp"
#include "PyBool.hpp"
#include "PyDict.hpp"
#include "PyFrozenSet.hpp"
#include "PyFunction.hpp"
#include "PyInteger.hpp"
#include "PyNone.hpp"
#include "PyNumber.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "StopIteration.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/NotImplementedError.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"

#include <algorithm>

namespace py {

template<> PySet *as(PyObject *obj)
{
	if (obj->type() == types::set()) { return static_cast<PySet *>(obj); }
	return nullptr;
}

template<> const PySet *as(const PyObject *obj)
{
	if (obj->type() == types::set()) { return static_cast<const PySet *>(obj); }
	return nullptr;
}

PySet::PySet() : PyBaseObject(types::BuiltinTypes::the().set()) {}

PySet::PySet(PyType *type) : PyBaseObject(type) {}

PySet::PySet(SetType elements) : PySet() { m_elements = std::move(elements); }

PyResult<PySet *> PySet::create(SetType elements)
{
	auto *result = VirtualMachine::the().heap().allocate<PySet>(elements);
	if (!result) { return Err(memory_error(sizeof(PySet))); }
	return Ok(result);
}

PyResult<PySet *> PySet::create()
{
	auto *result = VirtualMachine::the().heap().allocate<PySet>();
	if (!result) { return Err(memory_error(sizeof(PySet))); }
	return Ok(result);
}

PyResult<PyObject *> PySet::add(PyObject *element)
{
	m_elements.insert(element);
	return Ok(py_none());
}

PyResult<PyObject *> PySet::discard(PyObject *element)
{
	m_elements.erase(element);
	return Ok(py_none());
}

PyResult<PyObject *> PySet::remove(PyObject *element)
{
	if (m_elements.erase(element) == 1) {
		return Ok(py_none());
	} else {
		return element->repr().and_then([](PyObject *repr) -> PyResult<PyObject *> {
			return Err(key_error("{}", repr->to_string()));
		});
	}
}

PyResult<PySet *> PySet::update(PyObject *others)
{
	auto others_iterator = others->iter();
	if (others_iterator.is_err()) { return Err(others_iterator.unwrap_err()); }

	auto others_value = others_iterator.unwrap()->next();
	while (others_value.is_ok()) {
		m_elements.insert(others_value.unwrap());
		others_value = others_iterator.unwrap()->next();
	}

	if (!others_value.unwrap_err()->type()->issubclass(py::types::stop_iteration())) {
		return Err(others_value.unwrap_err());
	}

	return Ok(this);
}

PyResult<PySet *> PySet::intersection(PyTuple *args, PyDict *kwargs) const
{
	if (kwargs && kwargs->map().size() > 0) {
		return Err(type_error("intersection() takes no keyword arguments"));
	}

	SetType result;
	for (const auto &el_ : m_elements) {
		auto el_obj_ = PyObject::from(el_);
		if (el_obj_.is_err()) { return Err(el_obj_.unwrap_err()); }
		auto *el = el_obj_.unwrap();
		bool contains_value = true;
		for (size_t i = 1; i < args->elements().size(); ++i) {
			const auto &arg = args->elements()[i];
			auto obj_ = PyObject::from(arg);
			if (obj_.is_err()) { return Err(obj_.unwrap_err()); }
			auto contains_ = obj_.unwrap()->contains(el);
			if (contains_.is_err()) { return Err(contains_.unwrap_err()); }
			contains_value &= contains_.unwrap();
			if (!contains_value) { break; }
		}
		if (contains_value) { result.insert(el_); }
	}

	return PySet::create(result);
}

PyResult<PyObject *> PySet::pop()
{
	if (m_elements.empty()) { return Err(key_error("pop from an empty set")); }
	m_elements.erase(m_elements.begin());
	return Ok(py_none());
}

PyResult<PyObject *> PySet::issubset(const PyObject *other) const
{
	if (this == other) { return Ok(py_true()); }

	// fastpath
	if (other->type()->issubclass(types::set())) {
		const auto &other_set = static_cast<const PySet &>(*other);
		for (const auto &el : m_elements) {
			if (!other_set.elements().contains(el)) { return Ok(py_false()); }
		}
		return Ok(py_true());
	}
	return Err(not_implemented_error(
		"set.issubset not implemented when arg is of type {}", other->type()->to_string()));
}

std::string PySet::to_string() const
{
	std::ostringstream os;

	if (m_elements.empty()) {
		os << "set()";
	} else {
		os << "{";
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
		os << "}";
	}

	return os.str();
}

PyResult<PyObject *> PySet::__new__(const PyType *type, PyTuple *, PyDict *)
{
	ASSERT(type == types::set());

	return PySet::create();
}

PyResult<int32_t> PySet::__init__(PyTuple *args, PyDict *kwargs)
{
	if (kwargs && !kwargs->map().empty()) {
		return Err(type_error("set() takes no keyword arguments"));
	}
	if (!args || args->elements().size() == 0) { return Ok(0); }

	if (args->elements().size() != 1) {
		return Err(type_error("set expected at most 1 argument, got {}", args->elements().size()));
	}

	auto iterable = PyObject::from(args->elements()[0]);
	if (iterable.is_err()) return Err(iterable.unwrap_err());
	return from_iterable(iterable.unwrap(), std::inserter(m_elements, m_elements.begin()))
		.and_then([](auto) { return Ok(0); });
}

PyResult<PyObject *> PySet::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PySet::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	auto *it = heap.allocate<PySetIterator>(*this);
	if (!it) { return Err(memory_error(sizeof(PySetIterator))); }
	return Ok(it);
}

PyResult<size_t> PySet::__len__() const { return Ok(m_elements.size()); }

PyResult<PyObject *> PySet::__eq__(const PyObject *other) const
{
	if (!other->type()->issubclass(types::set())) { return Ok(py_false()); }
	return Ok(m_elements == static_cast<const PySet &>(*other).elements() ? py_true() : py_false());
}

PyResult<PyObject *> PySet::__le__(const PyObject *other) const
{
	if (!other->type()->issubclass(types::set())) {
		return Err(type_error(
			"'<=' not supported between instances of 'set' and '{}'", other->type()->to_string()));
	}
	return issubset(other);
}

PyResult<PyObject *> PySet::__lt__(const PyObject *other) const
{
	if (!other->type()->issubclass(types::set())) {
		return Err(type_error(
			"'<' not supported between instances of 'set' and '{}'", other->type()->to_string()));
	}
	return __eq__(other).and_then([this, other](PyObject *is_equal) -> PyResult<PyObject *> {
		if (is_equal == py_false()) { return issubset(other); }
		return Ok(py_false());
	});
}


PyResult<bool> PySet::__contains__(const PyObject *value) const
{
	const Value value_{ const_cast<PyObject *>(value) };
	return Ok(m_elements.contains(value_));
}

PyResult<PyObject *> PySet::__and__(PyObject *other)
{
	if (!other->type()->issubclass(types::set())) {
		return Err(
			type_error("unsupported operand type(s) for &: 'set' and '{}'", other->type()->name()));
	}
	const auto *other_set = static_cast<PySet *>(other);

	SetType result;
	std::copy_if(m_elements.begin(),
		m_elements.end(),
		std::inserter(result, result.begin()),
		[other_set](const auto &element) { return other_set->m_elements.contains(element); });

	return PySet::create(result);
}

void PySet::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto &el : m_elements) {
		if (std::holds_alternative<PyObject *>(el)) {
			if (std::get<PyObject *>(el) != this) visitor.visit(*std::get<PyObject *>(el));
		}
	}
}

PyType *PySet::static_type() const { return types::set(); }

namespace {

	std::once_flag set_flag;

	std::unique_ptr<TypePrototype> register_set()
	{
		return std::move(klass<PySet>("set")
							 .def("add", &PySet::add)
							 .def("discard", &PySet::discard)
							 .def("remove", &PySet::remove)
							 .def("intersection", &PySet::intersection)
							 .def("update", &PySet::update)
							 .def("pop", &PySet::pop)
							 .def("issubset", &PySet::issubset)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PySet::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(set_flag, []() { type = register_set(); });
		return std::move(type);
	};
}

PySetIterator::PySetIterator(PyType *type) : PyBaseObject(type) {}

PySetIterator::PySetIterator(const PySet &pyset)
	: PyBaseObject(types::BuiltinTypes::the().set_iterator()), m_pyset(pyset)
{}

PySetIterator::PySetIterator(const PyFrozenSet &pyset)
	: PyBaseObject(types::BuiltinTypes::the().set_iterator()), m_pyset(pyset)
{}

std::string PySetIterator::to_string() const
{
	return fmt::format("<set_iterator at {}>", static_cast<const void *>(this));
}

void PySetIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	// TODO: should visit_graph be const and the bit flags mutable?
	std::visit(
		[&visitor]<typename T>(const T &el) {
			if constexpr (!std::is_same_v<T, std::monostate>) {
				visitor.visit(const_cast<typename std::remove_cv_t<typename T::type> &>(el.get()));
			}
		},
		m_pyset);
}

PyResult<PyObject *> PySetIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PySetIterator::__next__()
{
	return std::visit(
		[&]<typename T>(const T &set_ref) -> PyResult<PyObject *> {
			if constexpr (!std::is_same_v<T, std::monostate>) {
				const auto &set = set_ref.get();
				if (m_current_index < set.elements().size()) {
					return std::visit([](const auto &element) { return PyObject::from(element); },
						*std::next(set.elements().begin(), m_current_index++));
				}
				return Err(stop_iteration());
			} else {
				TODO();
			}
		},
		m_pyset);
}

PyType *PySetIterator::static_type() const { return types::set_iterator(); }

namespace {

	std::once_flag set_iterator_flag;

	std::unique_ptr<TypePrototype> register_set_iterator()
	{
		return std::move(klass<PySetIterator>("set_iterator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PySetIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(set_iterator_flag, []() { type = register_set_iterator(); });
		return std::move(type);
	};
}

}// namespace py
