#include "PyList.hpp"
#include "MemoryError.hpp"
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

using namespace py;

template<> PyList *py::as(PyObject *obj)
{
	if (obj->type() == list()) { return static_cast<PyList *>(obj); }
	return nullptr;
}

template<> const PyList *py::as(const PyObject *obj)
{
	if (obj->type() == list()) { return static_cast<const PyList *>(obj); }
	return nullptr;
}

PyList::PyList() : PyBaseObject(BuiltinTypes::the().list()) {}

PyList::PyList(std::vector<Value> elements) : PyList() { m_elements = std::move(elements); }

PyResult PyList::create(std::vector<Value> elements)
{
	auto *result = VirtualMachine::the().heap().allocate<PyList>(elements);
	if (!result) { return PyResult::Err(memory_error(sizeof(PyList))); }
	return PyResult::Ok(result);
}

PyResult PyList::create()
{
	auto *result = VirtualMachine::the().heap().allocate<PyList>();
	if (!result) { return PyResult::Err(memory_error(sizeof(PyList))); }
	return PyResult::Ok(result);
}

PyResult PyList::append(PyTuple *args, PyDict *kwargs)
{
	ASSERT(args && args->size() == 1)
	ASSERT(!kwargs || kwargs->map().size())
	return PyObject::from(args->elements()[0]).and_then<PyObject>([this](auto *obj) {
		m_elements.push_back(obj);
		return PyResult::Ok(py_none());
	});
}

PyResult PyList::extend(PyTuple *args, PyDict *kwargs)
{
	ASSERT(args && args->size() == 1)
	ASSERT(!kwargs || kwargs->map().size())

	// FIXME: should check if object it iterable and the iterate
	return PyObject::from(args->elements()[0]).and_then<PyObject>([this](auto *iterable) {
		if (as<PyTuple>(iterable)) {
			for (const auto &el : as<PyTuple>(iterable)->elements()) { m_elements.push_back(el); }
		} else if (as<PyList>(iterable)) {
			for (const auto &el : as<PyList>(iterable)->elements()) { m_elements.push_back(el); }
		} else {
			TODO();
		}
		return PyResult::Ok(py_none());
	});
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

PyResult PyList::__repr__() const { return PyString::create(to_string()); }

PyResult PyList::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	auto *it = heap.allocate<PyListIterator>(*this);
	if (!it) { return PyResult::Err(memory_error(sizeof(PyListIterator))); }
	return PyResult::Ok(it);
}

PyResult PyList::__len__() const { return PyInteger::create(m_elements.size()); }

PyResult PyList::__eq__(const PyObject *other) const
{
	if (!as<PyList>(other)) { return PyResult::Ok(py_false()); }

	auto *other_list = as<PyList>(other);
	// Value contains PyObject* so we can't just compare vectors with std::vector::operator==
	// otherwise if we compare PyObject* with PyObject* we compare the pointers, rather
	// than PyObject::__eq__(const PyObject*)
	if (m_elements.size() != other_list->elements().size()) { return PyResult::Ok(py_false()); }
	auto &interpreter = VirtualMachine::the().interpreter();
	const bool result = std::equal(m_elements.begin(),
		m_elements.end(),
		other_list->elements().begin(),
		[&interpreter](const auto &lhs, const auto &rhs) -> bool {
			const auto &result = equals(lhs, rhs, interpreter);
			ASSERT(result.is_ok())
			auto is_true = truthy(result.unwrap(), interpreter);
			ASSERT(is_true.is_ok())
			if (std::holds_alternative<NameConstant>(is_true.unwrap())) {
				ASSERT(std::holds_alternative<bool>(std::get<NameConstant>(is_true.unwrap()).value))
				return std::holds_alternative<bool>(std::get<NameConstant>(is_true.unwrap()).value);
			} else if (std::holds_alternative<PyObject *>(is_true.unwrap())) {
				return is_true.template unwrap_as<PyObject>() == py_true();
			} else {
				TODO();
			}
		});
	return PyResult::Ok(result ? py_true() : py_false());
}

void PyList::sort()
{
	std::sort(m_elements.begin(), m_elements.end(), [](const Value &lhs, const Value &rhs) -> bool {
		if (auto cmp = less_than(lhs, rhs, VirtualMachine::the().interpreter()); cmp.is_ok()) {
			auto is_true = truthy(cmp.unwrap(), VirtualMachine::the().interpreter());
			ASSERT(is_true.is_ok())
			if (std::holds_alternative<NameConstant>(is_true.unwrap())) {
				ASSERT(std::holds_alternative<bool>(std::get<NameConstant>(is_true.unwrap()).value))
				return std::holds_alternative<bool>(std::get<NameConstant>(is_true.unwrap()).value);
			} else if (std::holds_alternative<PyObject *>(is_true.unwrap())) {
				return is_true.template unwrap_as<PyObject>() == py_false();
			} else {
				TODO();
			}
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
						 .def("extend", &PyList::extend)
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

PyResult PyListIterator::__repr__() const { return PyString::create(to_string()); }

PyResult PyListIterator::__next__()
{
	if (m_current_index < m_pylist.elements().size())
		return std::visit([](const auto &element) { return PyObject::from(element); },
			m_pylist.elements()[m_current_index++]);
	return PyResult::Err(stop_iteration(""));
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
