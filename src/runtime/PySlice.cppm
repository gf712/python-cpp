module;
#include "memory/allocate.hpp"

export module py.runtime:slice;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyDict;
class PyType;

class PySlice : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;

  public:
	PyObject *m_start{ nullptr };
	PyObject *m_stop{ nullptr };
	PyObject *m_step{ nullptr };

  private:
	PySlice(PyType *);

	PySlice();
	PySlice(PyObject *stop);
	PySlice(PyObject *start, PyObject *stop, PyObject *end);

  protected:
	void visit_graph(Visitor &) override;

  public:
	static PyResult<PySlice *> create(std::int64_t stop);
	static PyResult<PySlice *> create(std::int64_t start, std::int64_t stop, std::int64_t end);

	static PyResult<PySlice *> create(PyObject *stop);
	static PyResult<PySlice *> create(PyObject *start, PyObject *stop, PyObject *end);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<std::int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __repr__() const;
	PyResult<std::int64_t> __hash__() const;
	PyResult<PyObject *> __eq__(const PyObject *obj) const;
	PyResult<PyObject *> __lt__(const PyObject *obj) const;

	PyResult<std::tuple<std::int64_t, std::int64_t, std::int64_t>> get_indices(
		std::int64_t length) const;

	PyResult<std::tuple<std::int64_t, std::int64_t, std::int64_t>> unpack() const;
	static std::tuple<std::int64_t, std::int64_t, std::int64_t> adjust_indices(std::int64_t start,
		std::int64_t stop,
		std::int64_t step,
		std::int64_t length);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
	std::string to_string() const override;
};

}// namespace py
