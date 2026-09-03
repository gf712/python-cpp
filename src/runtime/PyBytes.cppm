module;
#include "memory/allocate.hpp"

export module py.runtime:bytes;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyDict;
class PyType;

class PyBytes : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	Bytes m_value;

	PyBytes(PyType *);

  public:
	static PyResult<PyBytes *> create(Bytes number);
	static PyResult<PyBytes *> create();

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<std::int32_t> __init__(PyTuple *args, PyDict *kwargs);

	~PyBytes() = default;
	std::string to_string() const override;

	PyResult<PyObject *> __add__(const PyObject *obj) const;
	PyResult<PyObject *> __mul__(const PyObject *obj) const;
	PyResult<std::size_t> __len__() const;
	PyResult<PyObject *> __eq__(const PyObject *obj) const;
	PyResult<PyObject *> __iter__() const;

	PyResult<PyObject *> __getitem__(std::int64_t index);

	PyResult<PyObject *> __getitem__(PyObject *index);
	// PyResult<std::monostate> __setitem__(PyObject *index, PyObject *value);

	PyResult<PyObject *> __repr__() const;
	PyResult<std::int64_t> __hash__() const;

	PyResult<std::monostate> __getbuffer__(PyBuffer &, int);

	const Bytes &value() const { return m_value; }

	PyResult<PyObject *> decode(const std::string &encoding, const std::string &errors) const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	PyBytes();
	PyBytes(Bytes number);
};

class PyBytesIterator : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	PyBytes *m_bytes{ nullptr };
	std::size_t m_index{ 0 };

	PyBytesIterator(PyType *);

  public:
	static PyResult<PyBytesIterator *> create(PyBytes *bytes);
	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	void visit_graph(Visitor &) override;

  private:
	PyBytesIterator(PyBytes *bytes, std::size_t index);
};

}// namespace py
