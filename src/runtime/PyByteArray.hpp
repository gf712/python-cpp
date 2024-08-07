#pragma once

#include "PyObject.hpp"

namespace py {

class PyByteArray : public PyBaseObject
{
	friend class ::Heap;

	Bytes m_value;

	PyByteArray(PyType *);

  public:
	static PyResult<PyByteArray *> create(const Bytes &bytes);
	static PyResult<PyByteArray *> create();

	~PyByteArray() = default;
	std::string to_string() const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __repr__() const;

	PyResult<PyObject *> __getitem__(int64_t index);
	PyResult<std::monostate> __setitem__(int64_t index, PyObject *value);

	const Bytes &value() const { return m_value; }

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	PyByteArray(const Bytes &value);
};

class PyByteArrayIterator : public PyBaseObject
{
	friend class ::Heap;

	PyByteArray *m_bytes{ nullptr };
	size_t m_index{ 0 };

	PyByteArrayIterator(PyType *);

  public:
	static PyResult<PyByteArrayIterator *> create(PyByteArray *bytes_array);
	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	void visit_graph(Visitor &) override;

  private:
	PyByteArrayIterator(PyByteArray *bytes, size_t index);
};

}// namespace py
