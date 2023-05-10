#include "PyMap.hpp"
#include "PyList.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

PyMap::PyMap(PyType *type) : PyBaseObject(type) {}

PyMap::PyMap(PyType *type, PyObject *func, PyTuple *iters)
	: PyBaseObject(type), m_func(func), m_iters(iters)
{}

std::string PyMap::to_string() const
{
	return fmt::format("<map object at {}>", static_cast<const void *>(this));
}


PyResult<PyObject *> PyMap::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	if (type == map() && kwargs && !kwargs->map().empty()) {
		return Err(type_error("map() takes no keyword arguments"));
	}

	if (!args || args->size() < 2) {
		return Err(type_error("map() must have at least two arguments."));
	}

	auto func_ = PyObject::from(args->elements()[0]);
	if (func_.is_err()) { return func_; }
	auto *func = func_.unwrap();

	auto iters_list_ = PyList::create();
	if (iters_list_.is_err()) { return iters_list_; }
	auto *iters_list = iters_list_.unwrap();

	for (size_t i = 1; i < args->size(); ++i) {
		auto iter_ = PyObject::from(args->elements()[i]).and_then([](PyObject *iterable) {
			return iterable->iter();
		});
		if (iter_.is_err()) { return iter_; }
		auto *iter = iter_.unwrap();
		iters_list->elements().push_back(iter);
	}

	auto iters_ = PyTuple::create(iters_list->elements());
	if (iters_.is_err()) { return iters_; }
	auto *iters = iters_.unwrap();

	auto obj =
		VirtualMachine::the().heap().allocate<PyMap>(const_cast<PyType *>(type), func, iters);
	if (!obj) { return Err(memory_error(sizeof(PyMap))); }
	return Ok(obj);
}

PyResult<PyObject *> PyMap::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyMap::__iter__() const { return Ok(const_cast<PyMap *>(this)); }

PyResult<PyObject *> PyMap::__next__()
{
	auto args_ = PyList::create();
	if (args_.is_err()) { return args_; }
	auto *args = args_.unwrap();

	ASSERT(m_iters);
	for (const auto &el : m_iters->elements()) {
		auto iter_ = PyObject::from(el);
		if (iter_.is_err()) { return iter_; }
		auto *iter = iter_.unwrap();
		auto value = iter->next();
		if (value.is_err()) { return value; }
		args->elements().push_back(value.unwrap());
	}

	auto args_tuple = PyTuple::create(args->elements());
	if (args_tuple.is_err()) { return args_tuple; }
	return m_func->call(args_tuple.unwrap(), nullptr);
}

namespace {

	std::once_flag map_flag;

	std::unique_ptr<TypePrototype> register_map() { return std::move(klass<PyMap>("map").type); }
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyMap::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(map_flag, []() { type = register_map(); });
		return std::move(type);
	};
}

PyType *PyMap::static_type() const { return map(); }

void PyMap::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_func) { visitor.visit(*m_func); }
	if (m_iters) { visitor.visit(*m_iters); }
}


}// namespace py
