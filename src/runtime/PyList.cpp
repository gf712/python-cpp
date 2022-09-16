#include "PyList.hpp"
#include "MemoryError.hpp"
#include "PyBool.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyGenericAlias.hpp"
#include "PyInteger.hpp"
#include "PyNone.hpp"
#include "PyNumber.hpp"
#include "PySlice.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "StopIteration.hpp"
#include "ValueError.hpp"
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

PyResult<PyList *> PyList::create(std::vector<Value> elements)
{
	auto *result = VirtualMachine::the().heap().allocate<PyList>(elements);
	if (!result) { return Err(memory_error(sizeof(PyList))); }
	return Ok(result);
}

PyResult<PyList *> PyList::create()
{
	auto *result = VirtualMachine::the().heap().allocate<PyList>();
	if (!result) { return Err(memory_error(sizeof(PyList))); }
	return Ok(result);
}

PyResult<PyObject *> PyList::append(PyObject *element)
{
	m_elements.push_back(element);
	return Ok(py_none());
}

PyResult<PyObject *> PyList::extend(PyObject *iterable)
{
	auto iterator = iterable->iter();
	if (iterator.is_err()) return iterator;

	auto tmp_list = PyList::create().unwrap();
	auto value = iterator.unwrap()->next();
	while (value.is_ok()) {
		tmp_list->elements().push_back(value.unwrap());
		value = iterator.unwrap()->next();
	}

	if (!value.unwrap_err()->type()->issubclass(stop_iteration()->type())) { return value; }

	m_elements.insert(m_elements.end(), tmp_list->elements().begin(), tmp_list->elements().end());

	return Ok(py_none());
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

PyResult<PyObject *> PyList::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyList::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	auto *it = heap.allocate<PyListIterator>(*this);
	if (!it) { return Err(memory_error(sizeof(PyListIterator))); }
	return Ok(it);
}

PyResult<PyObject *> PyList::__getitem__(PyObject *index)
{
	if (auto index_int = as<PyInteger>(index)) {
		const auto i = index_int->as_i64();
		if (i >= 0) {
			if (static_cast<size_t>(i) >= m_elements.size()) {
				// FIXME: should be IndexError
				return Err(value_error("list index out of range"));
			}
			return PyObject::from(m_elements[i]);
		} else {
			TODO();
		}
	} else if (auto slice = as<PySlice>(index)) {
		auto indices_ = slice->unpack();
		if (indices_.is_err()) return Err(indices_.unwrap_err());
		const auto [start_, end_, step] = indices_.unwrap();

		const auto [start, end, slice_length] =
			PySlice::adjust_indices(start_, end_, step, m_elements.size());

		if (slice_length == 0) { return PyList::create(); }
		if (start == 0 && end == static_cast<int64_t>(m_elements.size()) && step == 1) {
			return Ok(this);
		}

		auto new_list = PyList::create();
		if (new_list.is_err()) return new_list;

		for (int64_t idx = start, i = 0; i < slice_length; idx += step, ++i) {
			new_list.unwrap()->elements().push_back(m_elements[idx]);
		}
		return new_list;
	} else {
		return Err(
			type_error("list indices must be integers or slices, not {}", index->type()->name()));
	}
}

PyResult<size_t> PyList::__len__() const { return Ok(m_elements.size()); }

PyResult<PyObject *> PyList::__eq__(const PyObject *other) const
{
	if (!as<PyList>(other)) { return Ok(py_false()); }

	auto *other_list = as<PyList>(other);
	// Value contains PyObject* so we can't just compare vectors with std::vector::operator==
	// otherwise if we compare PyObject* with PyObject* we compare the pointers, rather
	// than PyObject::__eq__(const PyObject*)
	if (m_elements.size() != other_list->elements().size()) { return Ok(py_false()); }
	auto &interpreter = VirtualMachine::the().interpreter();
	const bool result = std::equal(m_elements.begin(),
		m_elements.end(),
		other_list->elements().begin(),
		[&interpreter](const auto &lhs, const auto &rhs) -> bool {
			const auto &result = equals(lhs, rhs, interpreter);
			ASSERT(result.is_ok())
			auto is_true = truthy(result.unwrap(), interpreter);
			ASSERT(is_true.is_ok())
			return is_true.unwrap();
		});
	return Ok(result ? py_true() : py_false());
}

void PyList::sort()
{
	std::sort(m_elements.begin(), m_elements.end(), [](const Value &lhs, const Value &rhs) -> bool {
		if (auto cmp = less_than(lhs, rhs, VirtualMachine::the().interpreter()); cmp.is_ok()) {
			auto is_true = truthy(cmp.unwrap(), VirtualMachine::the().interpreter());
			ASSERT(is_true.is_ok())
			return is_true.unwrap();
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
	return std::move(
		klass<PyList>("list")
			.def("append", &PyList::append)
			.def("extend", &PyList::extend)
			//  .def(
			// 	 "sort",
			// 	 +[](PyObject *self) {
			// 		 self->sort();
			// 		 return py_none();
			// 	 })
			.classmethod(
				"__class_getitem__",
				+[](PyType *type, PyTuple *args, PyDict *kwargs) {
					ASSERT(args && args->elements().size() == 1);
					ASSERT(!kwargs || kwargs->map().empty());
					return PyObject::from(args->elements()[0]).and_then([type](PyObject *arg) {
						return PyGenericAlias::create(type, arg);
					});
				})
			.type);
}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyList::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(list_flag, []() { type = ::register_list(); });
		return std::move(type);
	};
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

PyResult<PyObject *> PyListIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyListIterator::__next__()
{
	if (m_current_index < m_pylist.elements().size())
		return std::visit([](const auto &element) { return PyObject::from(element); },
			m_pylist.elements()[m_current_index++]);
	return Err(stop_iteration());
}

PyType *PyListIterator::type() const { return list_iterator(); }

namespace {

std::once_flag list_iterator_flag;

std::unique_ptr<TypePrototype> register_list_iterator()
{
	return std::move(klass<PyListIterator>("list_iterator").type);
}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyListIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(list_iterator_flag, []() { type = ::register_list_iterator(); });
		return std::move(type);
	};
}
