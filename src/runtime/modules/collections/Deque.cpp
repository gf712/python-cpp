#include "Deque.hpp"
#include "runtime/IndexError.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/Value.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <sstream>
#include <unordered_set>
#include <variant>

using namespace py;
using namespace py::collections;

namespace {
PyType *s_collections_deque = nullptr;

static std::unordered_set<PyObject *> visited;

}// namespace

Deque::Deque(PyType *type) : PyBaseObject(type) {}

PyResult<PyObject *>
	Deque::create(PyType *type, std::deque<Value> deque, std::optional<size_t> maxlength)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<Deque>(const_cast<PyType *>(type));
	if (!result) { return Err(memory_error(sizeof(BaseException))); }

	result->m_deque = std::move(deque);
	result->m_maxlength = std::move(maxlength);

	return Ok(result);
}


PyResult<PyObject *> Deque::__new__(const PyType *type, PyTuple *, PyDict *)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<Deque>(const_cast<PyType *>(type));
	if (!result) { return Err(memory_error(sizeof(BaseException))); }
	return Ok(result);
}

PyResult<int32_t> Deque::__init__(PyTuple *args, PyDict *kwargs)
{
	PyObject *iterable = nullptr;
	std::optional<size_t> maxlen;
	if (args && args->elements().size() <= 2) {
		if (args->elements().size() > 0) {
			auto el0 = PyObject::from(args->elements()[0]);
			if (el0.is_err()) { return Err(el0.unwrap_err()); }
			iterable = el0.unwrap();
		}
		if (args->elements().size() > 1) {
			auto el1 = PyObject::from(args->elements()[1]);
			if (el1.is_err()) { return Err(el1.unwrap_err()); }
			if (el1.unwrap()->type()->issubclass(types::integer())) {
				auto max_len = static_cast<const PyInteger &>(*el1.unwrap()).as_big_int();
				if (max_len < 0) { return Err(type_error("maxlen must be non-negative")); }
				ASSERT(max_len.fits_uint_p());
				maxlen = max_len.get_ui();
			} else if (el1.unwrap() != py_none()) {
				return Err(type_error("an integer is required"));
			}
		}
	}
	if (kwargs && !kwargs->map().empty()) { TODO(); }
	m_maxlength = maxlen;
	if (!iterable) { return Ok(0); }
	return extend(iterable).and_then([](auto) -> PyResult<int32_t> { return Ok(0); });
}

PyResult<PyObject *> Deque::__repr__() const
{
	std::ostringstream os;

	[[maybe_unused]] struct Cleanup
	{
		const Deque *d;
		bool do_cleanup;

		~Cleanup()
		{
			if (do_cleanup) {
				auto it = visited.find(const_cast<Deque *>(d));
				if (it != visited.end()) { visited.erase(it); }
			}
		}
	} cleanup{ this, !visited.contains(const_cast<Deque *>(this)) };
	visited.insert(const_cast<Deque *>(this));

	auto repr = [](const auto &el) -> PyResult<PyString *> {
		return std::visit(overloaded{
							  [](const auto &value) { return PyString::create(value.to_string()); },
							  [](PyObject *value) {
								  if (visited.contains(value)) { return PyString::create("[...]"); }
								  return value->repr();
							  },
						  },
			el);
	};
	os << type()->name() << "([";
	if (!m_deque.empty()) {
		auto it = m_deque.begin();
		while (std::next(it) != m_deque.end()) {
			auto r = repr(*it);
			if (r.is_err()) { return r; }
			os << std::move(r.unwrap()->value()) << ", ";
			std::advance(it, 1);
		}
		auto r = repr(*it);
		if (r.is_err()) { return r; }
		os << std::move(r.unwrap()->value());
	}
	os << "])";

	return PyString::create(os.str());
}


PyResult<PyObject *> Deque::append(PyObject *x)
{
	push_back(x);
	return Ok(py_none());
}

PyResult<PyObject *> Deque::appendleft(PyObject *x)
{
	push_front(x);
	return Ok(py_none());
}

PyResult<PyObject *> Deque::clear()
{
	m_deque.clear();
	return Ok(py_none());
}

PyResult<PyObject *> Deque::copy() { return create(type(), m_deque, m_maxlength); }

PyResult<PyObject *> Deque::count(PyObject *x)
{
	size_t count = 0;
	for (const auto &el : m_deque) {
		auto equals_ = equals(x, el, VirtualMachine::the().interpreter()).and_then([](auto cmp) {
			return truthy(cmp, VirtualMachine::the().interpreter());
		});

		if (equals_.is_err()) { return Err(equals_.unwrap_err()); }

		if (equals_.unwrap()) { count++; }
	}

	return PyInteger::create(count);
}

PyResult<PyObject *> Deque::extend(PyObject *iterable)
{
	auto iter_ = iterable->iter();
	if (iter_.is_err()) { return iter_; }
	auto *iter = iter_.unwrap();

	auto value_ = iter->next();
	while (value_.is_ok()) {
		push_back(value_.unwrap());
		value_ = iter->next();
	}

	if (!value_.unwrap_err()->type()->issubclass(types::stop_iteration())) { return value_; }

	return Ok(py_none());
}


PyResult<PyObject *> Deque::extendleft(PyObject *iterable)
{
	auto iter_ = iterable->iter();
	if (iter_.is_err()) { return iter_; }
	auto *iter = iter_.unwrap();

	auto value_ = iter->next();
	while (value_.is_ok()) {
		push_front(value_.unwrap());
		value_ = iter->next();
	}

	if (!value_.unwrap_err()->type()->issubclass(types::stop_iteration())) { return value_; }

	return Ok(py_none());
}

PyResult<PyObject *> Deque::pop()
{
	if (m_deque.empty()) { return Err(index_error("pop from an empty deque")); }
	auto result = m_deque.back();
	m_deque.pop_back();
	return PyObject::from(result);
}

PyResult<PyObject *> Deque::popleft()
{
	if (m_deque.empty()) { return Err(index_error("pop from an empty deque")); }
	auto result = m_deque.front();
	m_deque.pop_front();
	return PyObject::from(result);
}

PyResult<PyObject *> Deque::remove(PyObject *value)
{
	for (size_t i = 0; i < m_deque.size(); ++i) {
		const auto &el = m_deque[i];
		auto equals_ =
			equals(value, el, VirtualMachine::the().interpreter()).and_then([](auto cmp) {
				return truthy(cmp, VirtualMachine::the().interpreter());
			});

		if (equals_.is_err()) { return Err(equals_.unwrap_err()); }

		if (equals_.unwrap()) {
			m_deque.erase(m_deque.begin() + i);
			return Ok(py_none());
		}
	}

	return Err(value_error("{} is not in deque", value->to_string()));
}

PyResult<PyObject *> Deque::reverse()
{
	std::reverse(m_deque.begin(), m_deque.end());
	return Ok(py_none());
}

PyResult<PyObject *> Deque::rotate(PyObject *n)
{
	if (!n->type()->issubclass(types::integer())) {
		return Err(
			type_error("'{}' object cannot be interpreted as an integer", n->type()->name()));
	}

	ASSERT(static_cast<const PyInteger &>(*n).as_big_int().fits_slong_p());
	const auto count = static_cast<const PyInteger &>(*n).as_big_int().get_si();

	if (count < 0) {
		auto middle = std::abs(count) % m_deque.size();
		std::rotate(m_deque.begin(), m_deque.begin() + middle, m_deque.end());
	} else if (count > 0) {
		auto middle = count % m_deque.size();
		std::rotate(m_deque.rbegin(), m_deque.rbegin() + middle, m_deque.rend());
	}


	return Ok(py_none());
}

void Deque::push_back(Value value)
{
	ASSERT(m_deque.size() <= m_maxlength.value_or(-1));

	if (m_maxlength.has_value() && m_deque.size() == *m_maxlength) { m_deque.pop_front(); }

	m_deque.push_back(std::move(value));
}

PyResult<size_t> Deque::__len__() const { return Ok(m_deque.size()); }

PyResult<PyObject *> Deque::__getitem__(int64_t index) const
{
	if (index < 0) { index += m_deque.size(); }

	if (index < 0 || index >= static_cast<int64_t>(m_deque.size())) {
		return Err(index_error("deque index out of range"));
	}

	return PyObject::from(m_deque[index]);
}

void Deque::push_front(Value value)
{
	ASSERT(m_deque.size() <= m_maxlength.value_or(-1));

	if (m_maxlength.has_value() && m_deque.size() == *m_maxlength) { m_deque.pop_back(); }

	m_deque.push_front(std::move(value));
}

void Deque::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto &el : m_deque) {
		if (std::holds_alternative<PyObject *>(el)) {
			if (std::get<PyObject *>(el)) { visitor.visit(*std::get<PyObject *>(el)); }
		}
	}
}

PyType *Deque::register_type(PyModule *module)
{
	if (!s_collections_deque) {
		s_collections_deque = klass<Deque>(module, "collections.deque")
								  .def("append", &Deque::append)
								  .def("appendleft", &Deque::appendleft)
								  .def("clear", &Deque::clear)
								  .def("copy", &Deque::copy)
								  .def("count", &Deque::count)
								  .def("extend", &Deque::extend)
								  .def("extendleft", &Deque::extendleft)
								  .def("pop", &Deque::pop)
								  .def("popleft", &Deque::popleft)
								  .def("remove", &Deque::remove)
								  .def("reverse", &Deque::reverse)
								  .def("rotate", &Deque::rotate)
								  .finalize();
	}
	return s_collections_deque;
}