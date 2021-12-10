#pragma once

#include "PyObject.hpp"


class PyBytes : public PyBaseObject
{
	friend class Heap;

	Bytes m_value;

  public:
	static PyBytes *create(const Bytes &number);
	~PyBytes() = default;
	std::string to_string() const override;

	PyObject *__add__(const PyObject *obj) const;

	const Bytes &value() const { return m_value; }

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

  private:
	PyBytes(const Bytes &number);
};
