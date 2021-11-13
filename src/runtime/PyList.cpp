#include "PyList.hpp"
#include "PyDict.hpp"
#include "PyFunction.hpp"
#include "PyNumber.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "StopIterationException.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

PyList::PyList() : PyObject(PyObjectType::PY_LIST)
{
	put("append",
		VirtualMachine::the().heap().allocate<PyNativeFunction>(
			"append", [](PyTuple *args, PyDict *) {
				ASSERT(args->elements().size() == 2)
				auto this_obj = args->elements()[0];
				ASSERT(std::holds_alternative<PyObject *>(this_obj))
				ASSERT(as<PyList>(std::get<PyObject *>(this_obj)))
				return as<PyList>(std::get<PyObject *>(this_obj))
					->append(PyObject::from(args->elements()[1]));
			}));
}

PyList::PyList(std::vector<Value> elements) : PyList() { m_elements = std::move(elements); }

PyList *PyList::create(std::vector<Value> elements)
{
	return VirtualMachine::the().heap().allocate<PyList>(elements);
}

PyList *PyList::create() { return VirtualMachine::the().heap().allocate<PyList>(); }

PyObject *PyList::append(PyObject *obj)
{
	m_elements.push_back(obj);
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

PyObject *PyList::repr_impl(Interpreter &) const { return PyString::from(String{ to_string() }); }

PyObject *PyList::iter_impl(Interpreter &) const
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyListIterator>(*this);
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

PyObject *PyListIterator::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

PyObject *PyListIterator::next_impl(Interpreter &interpreter)
{
	if (m_current_index < m_pylist.elements().size())
		return std::visit([](const auto &element) { return PyObject::from(element); },
			m_pylist.elements()[m_current_index++]);
	interpreter.raise_exception(stop_iteration(""));
	return nullptr;
}
