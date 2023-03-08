#include "PyZip.hpp"
#include "MemoryError.hpp"
#include "PyList.hpp"
#include "StopIteration.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

PyZip::PyZip(PyType *type) : PyBaseObject(type) {}

PyZip::PyZip(std::vector<PyObject *> &&iterators)
	: PyBaseObject(BuiltinTypes::the().zip()), m_iterators(std::move(iterators))
{}

PyResult<PyObject *> PyZip::create(PyTuple *iterables)
{
	auto *result = VirtualMachine::the().heap().allocate<PyZip>(std::vector<PyObject *>{});
	if (!result) { return Err(memory_error(sizeof(PyZip))); }

	// result keeps track of the iterators during GC
	for (const auto &iterable_ : iterables->elements()) {
		auto iterable = PyObject::from(iterable_);
		if (iterable.is_err()) { return iterable; }
		auto iterator = iterable.unwrap()->iter();
		if (iterator.is_err()) { return iterator; }
		result->m_iterators.push_back(iterator.unwrap());
	}

	return Ok(result);
}

PyResult<PyObject *> PyZip::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == zip());
	ASSERT(!kwargs || kwargs->map().empty());
	return PyZip::create(args);
}

PyResult<PyObject *> PyZip::__iter__() const { return Ok(const_cast<PyZip *>(this)); }

PyResult<PyObject *> PyZip::__next__()
{
	if (m_iterators.empty()) {
		return Err(stop_iteration());
	} else {
		// use a list here because it is mutable
		auto els = PyList::create();
		if (els.is_err()) { return els; }
		for (auto *it : m_iterators) {
			if (auto value = it->next(); value.is_ok()) {
				els.unwrap()->elements().push_back(value.unwrap());
			} else {
				return value;
			}
		}
		return PyTuple::create(els.unwrap()->elements());
		;
	}
}

PyType *PyZip::static_type() const { return zip(); }

void PyZip::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	for (auto *it : m_iterators) {
		ASSERT(it);
		visitor.visit(*it);
	}
}

namespace {

	std::once_flag zip_flag;

	std::unique_ptr<TypePrototype> zip_reversed() { return std::move(klass<PyZip>("zip").type); }
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyZip::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(zip_flag, []() { type = zip_reversed(); });
		return std::move(type);
	};
}

}// namespace py
