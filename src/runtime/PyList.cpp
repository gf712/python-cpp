#include "PyList.hpp"
#include "PyBool.hpp"
#include "PyDict.hpp"
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
#include "vm/VM.hpp"

template<> PyList *as(PyObject *obj)
{
	if (obj->type() == list()) { return static_cast<PyList *>(obj); }
	return nullptr;
}

template<> const PyList *as(const PyObject *obj)
{
	if (obj->type() == list()) { return static_cast<const PyList *>(obj); }
	return nullptr;
}

PyList::PyList() : PyBaseObject(BuiltinTypes::the().list()) {}

PyList::PyList(std::vector<Value> elements) : PyList() { m_elements = std::move(elements); }

PyList *PyList::create(std::vector<Value> elements)
{
	return VirtualMachine::the().heap().allocate<PyList>(elements);
}

PyList *PyList::create() { return VirtualMachine::the().heap().allocate<PyList>(); }

PyObject *PyList::append(PyTuple *args, PyDict *kwargs)
{
	ASSERT(args && args->size() == 1)
	ASSERT(!kwargs || kwargs->map().size())
	m_elements.push_back(PyObject::from(args->elements()[0]));
	return py_none();
}

std::string PyList::to_string() const
{
	std::ostringstream os;

	os << "[";
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
	os << "]";

	return os.str();
}

PyObject *PyList::__repr__() const { return PyString::from(String{ to_string() }); }

PyObject *PyList::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyListIterator>(*this);
}

PyObject *PyList::__len__() const { return PyInteger::create(m_elements.size()); }

PyObject *PyList::__eq__(const PyObject *other) const
{
	if (!as<PyList>(other)) { return py_false(); }

	auto *other_list = as<PyList>(other);
	// Value contains PyObject* so we can't just compare vectors with std::vector::operator==
	// otherwise if we compare PyObject* with PyObject* we compare the pointers, rather
	// than PyObject::__eq__(const PyObject*)
	if (m_elements.size() != other_list->elements().size()) { return py_false(); }
	auto &interpreter = VirtualMachine::the().interpreter();
	const bool result = std::equal(m_elements.begin(),
		m_elements.end(),
		other_list->elements().begin(),
		[&interpreter](const auto &lhs, const auto &rhs) {
			const auto &result = equals(lhs, rhs, interpreter);
			ASSERT(result.has_value())
			return truthy(*result, interpreter);
		});
	return result ? py_true() : py_false();
}

void PyList::sort()
{
	std::sort(m_elements.begin(), m_elements.end(), [](const Value &lhs, const Value &rhs) -> bool {
		if (auto cmp = less_than(lhs, rhs, VirtualMachine::the().interpreter())) {
			return ::truthy(*cmp, VirtualMachine::the().interpreter());
		} else {
			// VirtualMachine::the().interpreter().raise_exception("Failed to compare {} with {}",
			// 	PyObject::from(lhs)->to_string(),
			// 	PyObject::from(rhs)->to_string());
			return false;
		}
	});
}

void PyList::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto &el : m_elements) {
		if (std::holds_alternative<PyObject *>(el)) {
			if (std::get<PyObject *>(el) != this) std::get<PyObject *>(el)->visit_graph(visitor);
		}
	}
}

PyType *PyList::type() const { return list(); }

namespace {

std::once_flag list_flag;

std::unique_ptr<TypePrototype> register_list()
{
	return std::move(klass<PyList>("list")
						 .def("append", &PyList::append)
						 //  .def(
						 // 	 "sort",
						 // 	 +[](PyObject *self) {
						 // 		 self->sort();
						 // 		 return py_none();
						 // 	 })
						 .type);
}
}// namespace

std::unique_ptr<TypePrototype> PyList::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(list_flag, []() { type = ::register_list(); });
	return std::move(type);
}


PyListIterator::PyListIterator(const PyList &pylist)
	: PyBaseObject(BuiltinTypes::the().list_iterator()), m_pylist(pylist)
{}

std::string PyListIterator::to_string() const
{
	return fmt::format("<list_iterator at {}>", static_cast<const void *>(this));
}

void PyListIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	// the iterator has to keep a reference to the list
	// otherwise GC could clean up a temporary list in a loop
	// TODO: should visit_graph be const and the bit flags mutable?
	const_cast<PyList &>(m_pylist).visit_graph(visitor);
}

PyObject *PyListIterator::__repr__() const { return PyString::create(to_string()); }

PyObject *PyListIterator::__next__()
{
	if (m_current_index < m_pylist.elements().size())
		return std::visit([](const auto &element) { return PyObject::from(element); },
			m_pylist.elements()[m_current_index++]);
	VirtualMachine::the().interpreter().raise_exception(stop_iteration(""));
	return nullptr;
}

PyType *PyListIterator::type() const { return list_iterator(); }

namespace {

std::once_flag list_iterator_flag;

std::unique_ptr<TypePrototype> register_list_iterator()
{
	return std::move(klass<PyListIterator>("list_iterator").type);
}
}// namespace

std::unique_ptr<TypePrototype> PyListIterator::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(list_iterator_flag, []() { type = ::register_list_iterator(); });
	return std::move(type);
}
