#pragma once

#include "PyObject.hpp"


class PyEllipsis : public PyBaseObject
{
	friend class Heap;
	friend PyObject *py_ellipsis();

	static constexpr Ellipsis m_value{};

  public:
	std::string to_string() const override { return fmt::format("PyEllipsis"); }

	PyObject *add_impl(const PyObject *obj) const;

	const Ellipsis &value() const { return m_value; }

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type_() const override;

  private:
	static PyEllipsis *create();
	PyEllipsis();
};

PyObject *py_ellipsis();
