#pragma once

#include "PyObject.hpp"

namespace py {

class PyEllipsis : public PyBaseObject
{
	friend class ::Heap;
	friend PyObject *py_ellipsis();

	static constexpr Ellipsis m_value{};

	PyEllipsis(PyType *);

  public:
	std::string to_string() const override { return fmt::format("PyEllipsis"); }

	PyResult<PyObject *> __add__(const PyObject *obj) const;
	PyResult<PyObject *> __repr__() const;

	const Ellipsis &value() const { return m_value; }

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

  private:
	static PyResult<PyEllipsis *> create();
	PyEllipsis();
};

PyObject *py_ellipsis();

}// namespace py
