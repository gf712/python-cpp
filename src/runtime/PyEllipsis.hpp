#pragma once

#include "PyObject.hpp"

namespace py {

class PyEllipsis : public PyBaseObject
{
	friend class ::Heap;
	friend PyObject *py_ellipsis();

	static constexpr Ellipsis m_value{};

  public:
	std::string to_string() const override { return fmt::format("PyEllipsis"); }

	PyObject *__add__(const PyObject *obj) const;
	PyObject *__repr__() const;

	const Ellipsis &value() const { return m_value; }

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

  private:
	static PyEllipsis *create();
	PyEllipsis();
};

PyObject *py_ellipsis();

}// namespace py