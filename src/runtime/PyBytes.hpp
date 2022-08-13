#pragma once

#include "PyObject.hpp"

namespace py {

class PyBytes : public PyBaseObject
{
	friend class ::Heap;

	Bytes m_value;

  public:
	static PyResult<PyBytes *> create(const Bytes &number);
	static PyResult<PyBytes *> create();

	~PyBytes() = default;
	std::string to_string() const override;

	PyResult<PyObject *> __add__(const PyObject *obj) const;
	PyResult<size_t> __len__() const;
	PyResult<PyObject *> __eq__(const PyObject *obj) const;

	PyResult<PyObject *> __repr__() const;

	const Bytes &value() const { return m_value; }

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *type() const override;

  private:
	PyBytes();
	PyBytes(const Bytes &number);
};

}// namespace py