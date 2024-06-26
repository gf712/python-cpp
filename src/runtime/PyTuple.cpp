#include "PyTuple.hpp"
#include "IndexError.hpp"
#include "MemoryError.hpp"
#include "PyBool.hpp"
#include "PyInteger.hpp"
#include "PyList.hpp"
#include "PySlice.hpp"
#include "PyString.hpp"
#include "StopIteration.hpp"
#include "TypeError.hpp"
#include "ValueError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> PyTuple *as(PyObject *obj)
{
	if (obj->type() == types::tuple()) { return static_cast<PyTuple *>(obj); }
	return nullptr;
}

template<> const PyTuple *as(const PyObject *obj)
{
	if (obj->type() == types::tuple()) { return static_cast<const PyTuple *>(obj); }
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

	std::vector<Value> make_value_vector(std::vector<PyObject *> &&elements)
	{
		ASSERT(std::all_of(
			elements.begin(), elements.end(), [](const auto &el) { return el != nullptr; }));
		std::vector<Value> result;
		result.reserve(elements.size());
		result.insert(result.end(), elements.begin(), elements.end());
		return result;
	}
}// namespace

PyTuple::PyTuple(PyType *type) : PyBaseObject(type) {}

PyTuple::PyTuple(PyType *type, std::vector<Value> elements)
	: PyBaseObject(type), m_elements(std::move(elements))
{
	ASSERT(std::all_of(m_elements.begin(), m_elements.end(), [](const auto &el) {
		if (std::holds_alternative<PyObject *>(el)) return std::get<PyObject *>(el) != nullptr;
		return true;
	}));
}

PyTuple::PyTuple(std::vector<Value> &&elements)
	: PyBaseObject(types::BuiltinTypes::the().tuple()), m_elements(std::move(elements))
{
	ASSERT(std::all_of(m_elements.begin(), m_elements.end(), [](const auto &el) {
		if (std::holds_alternative<PyObject *>(el)) return std::get<PyObject *>(el) != nullptr;
		return true;
	}));
}

PyTuple::PyTuple() : PyTuple(std::vector<Value>{}) {}

PyTuple::PyTuple(const std::vector<PyObject *> &elements) : PyTuple(make_value_vector(elements)) {}

PyTuple::PyTuple(PyType *type, const std::vector<PyObject *> &elements)
	: PyTuple(type, make_value_vector(elements))
{}

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

PyResult<PyTuple *> PyTuple::create(PyType *type, std::vector<Value> elements)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyTuple>(type, std::move(elements))) { return Ok(obj); }
	return Err(memory_error(sizeof(PyTuple)));
}

PyResult<PyTuple *> PyTuple::create(PyType *type, const std::vector<PyObject *> &elements)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyTuple>(type, elements)) { return Ok(obj); }
	return Err(memory_error(sizeof(PyTuple)));
}

PyResult<PyTuple *> PyTuple::create(const std::vector<PyObject *> &elements)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyTuple>(elements)) { return Ok(obj); }
	return Err(memory_error(sizeof(PyTuple)));
}

PyResult<PyTuple *> PyTuple::create(std::vector<PyObject *> &&elements)
{
	auto &heap = VirtualMachine::the().heap();
	if (auto *obj = heap.allocate<PyTuple>(make_value_vector(std::move(elements)))) {
		return Ok(obj);
	}
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

PyResult<PyObject *> PyTuple::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	if (!type->issubclass(types::tuple())) {
		return Err(type_error(
			"tuple.__new__({}): {} is not a subtype of tuple", type->name(), type->name()));
	}

	auto result = PyArgsParser<PyObject *>::unpack_tuple(args,
		kwargs,
		"tuple",
		std::integral_constant<size_t, 0>{},
		std::integral_constant<size_t, 1>{},
		nullptr /* iterable */);

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [iterable] = result.unwrap();
	if (!iterable) {
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<PyTuple>(const_cast<PyType *>(type))) { return Ok(obj); }
		return Err(memory_error(sizeof(PyTuple)));
	}

	auto iterator_ = iterable->iter();
	if (iterator_.is_err()) { return iterator_; }
	auto iterator = iterator_.unwrap();

	auto els_ = PyList::create();
	if (els_.is_err()) { return Err(els_.unwrap_err()); }
	auto els = els_.unwrap();

	auto value = iterator->next();
	while (value.is_ok()) {
		els->elements().push_back(value.unwrap());
		value = iterator->next();
	}

	if (!value.unwrap_err()->type()->issubclass(stop_iteration()->type())) { return value; }

	return PyTuple::create(const_cast<PyType *>(type), els->elements());
}

PyResult<PyObject *> PyTuple::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyTuple::__iter__() const { return PyTupleIterator::create(*this); }

PyResult<size_t> PyTuple::__len__() const { return Ok(m_elements.size()); }

PyResult<PyObject *> PyTuple::__add__(const PyObject *other) const
{
	auto *b = as<PyTuple>(other);
	if (!b) {
		return Err(
			type_error("can only concatenate tuple (not \"{}\") to tuple", other->type()->name()));
	}
	if (m_elements.empty()) return Ok(const_cast<PyObject *>(other));
	std::vector<Value> elements = m_elements;
	elements.insert(elements.end(), b->elements().begin(), b->elements().end());
	return PyTuple::create(elements);
}

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

PyResult<PyObject *> PyTuple::__getitem__(int64_t index)
{
	if (index < 0) {
		if (static_cast<size_t>(std::abs(index)) > m_elements.size()) {
			return Err(index_error("tuple index out of range"));
		}
		index += m_elements.size();
	}
	ASSERT(index >= 0);
	if (static_cast<size_t>(index) >= m_elements.size()) {
		return Err(index_error("tuple index out of range"));
	}
	return PyObject::from(m_elements[index]);
}

PyResult<PyObject *> PyTuple::__getitem__(PyObject *index)
{
	if (auto index_int = as<PyInteger>(index)) {
		const auto i = index_int->as_i64();
		return __getitem__(i);
	} else if (auto slice = as<PySlice>(index)) {
		auto indices_ = slice->unpack();
		if (indices_.is_err()) return Err(indices_.unwrap_err());
		const auto [start_, end_, step] = indices_.unwrap();

		const auto [start, end, slice_length] =
			PySlice::adjust_indices(start_, end_, step, m_elements.size());

		if (slice_length == 0) { return PyTuple::create(); }
		if (start == 0 && end == static_cast<int64_t>(m_elements.size()) && step == 1) {
			return Ok(this);
		}

		std::vector<Value> new_tuple_values;
		new_tuple_values.reserve(slice_length);
		for (int64_t idx = start, i = 0; i < slice_length; idx += step, ++i) {
			new_tuple_values.push_back(m_elements[idx]);
		}
		return PyTuple::create(new_tuple_values);
	} else {
		return Err(
			type_error("tuple indices must be integers or slices, not {}", index->type()->name()));
	}
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
			if (std::get<PyObject *>(el) != this) { visitor.visit(*std::get<PyObject *>(el)); }
		}
	}
}

PyType *PyTuple::static_type() const { return types::tuple(); }

namespace {

	std::once_flag tuple_flag;

	std::unique_ptr<TypePrototype> register_tuple()
	{
		return std::move(klass<PyTuple>("tuple").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyTuple::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(tuple_flag, []() { type = register_tuple(); });
		return std::move(type);
	};
}


PyTupleIterator::PyTupleIterator(const PyTuple &pytuple)
	: PyBaseObject(types::BuiltinTypes::the().tuple_iterator()), m_pytuple(pytuple)
{}

PyTupleIterator::PyTupleIterator(const PyTuple &pytuple, size_t position) : PyTupleIterator(pytuple)
{
	m_current_index = position;
}

PyResult<PyTupleIterator *> PyTupleIterator::create(const PyTuple &pytuple)
{
	auto &heap = VirtualMachine::the().heap();
	auto *obj = heap.allocate<PyTupleIterator>(pytuple);
	if (!obj) return Err(memory_error(sizeof(PyTupleIterator)));
	return Ok(obj);
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
	return Err(stop_iteration());
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
	PyObject::visit_graph(visitor);
	visitor.visit(const_cast<PyTuple &>(m_pytuple));
}


PyType *PyTupleIterator::static_type() const { return types::tuple_iterator(); }

namespace {

	std::once_flag tuple_iterator_flag;

	std::unique_ptr<TypePrototype> register_tuple_iterator()
	{
		return std::move(klass<PyTupleIterator>("tuple_iterator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyTupleIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(tuple_iterator_flag, []() { type = register_tuple_iterator(); });
		return std::move(type);
	};
}

}// namespace py
