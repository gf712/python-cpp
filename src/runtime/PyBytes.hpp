#pragma once

#include "PyObject.hpp"

namespace py {

class PyBytes : public PyBaseObject
{
	friend class ::Heap;

	Bytes m_value;

	PyBytes(PyType *);

  public:
	static PyResult<PyBytes *> create(const Bytes &number);
	static PyResult<PyBytes *> create();

	~PyBytes() = default;
	std::string to_string() const override;

	PyResult<PyObject *> __add__(const PyObject *obj) const;
	PyResult<size_t> __len__() const;
	PyResult<PyObject *> __eq__(const PyObject *obj) const;
	PyResult<PyObject *> __iter__() const;

	PyResult<PyObject *> __repr__() const;

	const Bytes &value() const { return m_value; }

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	PyBytes();
	PyBytes(const Bytes &number);
};

class PyBytesIterator : public PyBaseObject
{
	friend class ::Heap;

	PyBytes *m_bytes{ nullptr };
	size_t m_index{ 0 };

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
	PyBytesIterator(PyBytes *bytes, size_t index);
};

}// namespace py
