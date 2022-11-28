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
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PySet *as(PyObject *obj)
{
	if (obj->type() == set()) { return static_cast<PySet *>(obj); }
	return nullptr;
}

template<> const PySet *as(const PyObject *obj)
{
	if (obj->type() == set()) { return static_cast<const PySet *>(obj); }
	return nullptr;
}

PySet::PySet() : PyBaseObject(BuiltinTypes::the().set()) {}

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
	ASSERT(type == set());

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
	(void)other;
	TODO();
	return Err(nullptr);
}

PyResult<bool> PySet::__contains__(const PyObject *value) const
{
	const Value value_{ const_cast<PyObject *>(value) };
	return Ok(m_elements.contains(value_));
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

PyType *PySet::type() const { return set(); }

namespace {

	std::once_flag set_flag;

	std::unique_ptr<TypePrototype> register_set()
	{
		return std::move(klass<PySet>("set")
							 .def("add", &PySet::add)
							 .def("discard", &PySet::discard)
							 .def("remove", &PySet::remove)
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


PySetIterator::PySetIterator(const PySet &pyset)
	: PyBaseObject(BuiltinTypes::the().set_iterator()), m_pyset(pyset)
{}

PySetIterator::PySetIterator(const PyFrozenSet &pyset)
	: PyBaseObject(BuiltinTypes::the().set_iterator()), m_pyset(pyset)
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
			const_cast<typename std::remove_cv_t<typename T::type> &>(el.get()).visit_graph(
				visitor);
		},
		m_pyset);
}

PyResult<PyObject *> PySetIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PySetIterator::__next__()
{
	return std::visit(
		[&](const auto &set_ref) -> PyResult<PyObject *> {
			const auto &set = set_ref.get();
			if (m_current_index < set.elements().size()) {
				return std::visit([](const auto &element) { return PyObject::from(element); },
					*std::next(set.elements().begin(), m_current_index++));
			}
			return Err(stop_iteration());
		},
		m_pyset);
}

PyType *PySetIterator::type() const { return set_iterator(); }

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
