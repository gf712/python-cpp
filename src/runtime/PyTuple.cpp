#include "PyTuple.hpp"
#include "MemoryError.hpp"
#include "PyBool.hpp"
#include "PyInteger.hpp"
#include "PyString.hpp"
#include "StopIteration.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyTuple *as(PyObject *obj)
{
	if (obj->type() == tuple()) { return static_cast<PyTuple *>(obj); }
	return nullptr;
}

template<> const PyTuple *as(const PyObject *obj)
{
	if (obj->type() == tuple()) { return static_cast<const PyTuple *>(obj); }
	return nullptr;
}

namespace {
	std::vector<Value> make_value_vector(const std::vector<PyObject *> &elements)
	{
		ASSERT(std::all_of(
			elements.begin(), elements.end(), [](const auto &el) { return el != nullptr; }));
		std::vector<Value> result;
		result.reserve(elements.size());
		result.insert(result.end(), elements.begin(), elements.end());
		return result;
	}
}// namespace

PyTuple::PyTuple(std::vector<Value> &&elements)
	: PyBaseObject(BuiltinTypes::the().tuple()), m_elements(std::move(elements))
{
	ASSERT(std::all_of(m_elements.begin(), m_elements.end(), [](const auto &el) {
		if (std::holds_alternative<PyObject *>(el)) return std::get<PyObject *>(el) != nullptr;
		return true;
	}));
}

PyTuple::PyTuple() : PyTuple(std::vector<Value>{}) {}

PyTuple::PyTuple(const std::vector<PyObject *> &elements) : PyTuple(make_value_vector(elements)) {}

PyResult<PyTuple *> PyTuple::create()
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyTuple>()) { return Ok(obj); }
	return Err(memory_error(sizeof(PyTuple)));
}

PyResult<PyTuple *> PyTuple::create(std::vector<Value> &&elements)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyTuple>(std::move(elements))) { return Ok(obj); }
	return Err(memory_error(sizeof(PyTuple)));
}

PyResult<PyTuple *> PyTuple::create(const std::vector<PyObject *> &elements)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyTuple>(elements)) { return Ok(obj); }
	return Err(memory_error(sizeof(PyTuple)));
}

std::string PyTuple::to_string() const
{
	std::ostringstream os;

	os << "(";
	if (!m_elements.empty()) {
		auto it = m_elements.begin();
		while (std::next(it) != m_elements.end()) {
			std::visit(overloaded{ [&os](const auto &value) { os << value; },
						   [&os](PyObject *value) {
							   auto r = value->repr();
							   ASSERT(r.is_ok())
							   os << r.unwrap()->to_string();
						   } },
				*it);
			std::advance(it, 1);
			os << ", ";
		}
		std::visit(overloaded{ [&os](const auto &value) { os << value; },
					   [&os](PyObject *value) {
						   auto r = value->repr();
						   ASSERT(r.is_ok())
						   os << r.unwrap()->to_string();
					   } },
			*it);
	}
	if (m_elements.size() == 1) { os << ','; }
	os << ")";

	return os.str();
}

PyResult<PyObject *> PyTuple::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyTuple::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<PyTupleIterator>(*this);
	if (!obj) return Err(memory_error(sizeof(PyTupleIterator)));
	return Ok(obj);
}

PyResult<size_t> PyTuple::__len__() const { return Ok(m_elements.size()); }

PyResult<PyObject *> PyTuple::__eq__(const PyObject *other) const
{
	if (!as<PyTuple>(other)) { return Ok(py_false()); }

	auto *other_tuple = as<PyTuple>(other);
	// Value contains PyObject* so we can't just compare vectors with std::vector::operator==
	// otherwise if we compare PyObject* with PyObject* we compare the pointers, rather
	// than PyObject::__eq__(const PyObject*)
	if (m_elements.size() != other_tuple->elements().size()) { return Ok(py_false()); }
	auto &interpreter = VirtualMachine::the().interpreter();
	const bool result = std::equal(m_elements.begin(),
		m_elements.end(),
		other_tuple->elements().begin(),
		[&interpreter](const auto &lhs, const auto &rhs) -> bool {
			const auto &result = equals(lhs, rhs, interpreter);
			ASSERT(result.is_ok())
			auto is_true = truthy(result.unwrap(), interpreter);
			ASSERT(is_true.is_ok())
			return is_true.unwrap();
		});
	return Ok(result ? py_true() : py_false());
}

PyTupleIterator PyTuple::begin() const { return PyTupleIterator(*this); }

PyTupleIterator PyTuple::end() const { return PyTupleIterator(*this, m_elements.size()); }

PyResult<PyObject *> PyTuple::operator[](size_t idx) const
{
	return std::visit([](const auto &value) { return PyObject::from(value); }, m_elements[idx]);
}

void PyTuple::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto &el : m_elements) {
		if (std::holds_alternative<PyObject *>(el)) {
			if (std::get<PyObject *>(el) != this) std::get<PyObject *>(el)->visit_graph(visitor);
		}
	}
}

PyType *PyTuple::type() const { return tuple(); }

namespace {

	std::once_flag tuple_flag;

	std::unique_ptr<TypePrototype> register_tuple()
	{
		return std::move(klass<PyTuple>("tuple").type);
	}
}// namespace

std::unique_ptr<TypePrototype> PyTuple::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(tuple_flag, []() { type = register_tuple(); });
	return std::move(type);
}


PyTupleIterator::PyTupleIterator(const PyTuple &pytuple)
	: PyBaseObject(BuiltinTypes::the().tuple_iterator()), m_pytuple(pytuple)
{}

PyTupleIterator::PyTupleIterator(const PyTuple &pytuple, size_t position) : PyTupleIterator(pytuple)
{
	m_current_index = position;
}

std::string PyTupleIterator::to_string() const
{
	return fmt::format("<tuple_iterator at {}>", static_cast<const void *>(this));
}

PyResult<PyObject *> PyTupleIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyTupleIterator::__next__()
{
	if (m_current_index < m_pytuple.elements().size())
		return std::visit([](const auto &element) { return PyObject::from(element); },
			m_pytuple.elements()[m_current_index++]);
	return Err(stop_iteration(""));
}

bool PyTupleIterator::operator==(const PyTupleIterator &other) const
{
	return &m_pytuple == &other.m_pytuple && m_current_index == other.m_current_index;
}

PyTupleIterator &PyTupleIterator::operator++()
{
	m_current_index++;
	return *this;
}

PyTupleIterator &PyTupleIterator::operator--()
{
	m_current_index--;
	return *this;
}

PyResult<PyObject *> PyTupleIterator::operator*() const
{
	return std::visit([](const auto &element) { return PyObject::from(element); },
		m_pytuple.elements()[m_current_index]);
}

void PyTupleIterator::visit_graph(Visitor &visitor)
{
	const_cast<PyTuple &>(m_pytuple).visit_graph(visitor);
}


PyType *PyTupleIterator::type() const { return tuple_iterator(); }

namespace {

	std::once_flag tuple_iterator_flag;

	std::unique_ptr<TypePrototype> register_tuple_iterator()
	{
		return std::move(klass<PyTupleIterator>("tuple_iterator").type);
	}
}// namespace

std::unique_ptr<TypePrototype> PyTupleIterator::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(tuple_iterator_flag, []() { type = register_tuple_iterator(); });
	return std::move(type);
}

}// namespace py