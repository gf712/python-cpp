#pragma once

#include "PyObject.hpp"
#include "PyTuple.hpp"

#include <optional>

namespace py {

class PyString : public PyBaseObject
{
	friend class ::Heap;
	std::string m_value;

  public:
	static PyResult create(const std::string &value);
	static PyResult create(PyString *self, PyTuple *args, PyDict *kwargs)
	{
		// FIXME with proper error handling
		ASSERT(self)
		ASSERT(!args || (args->size() == 0))
		ASSERT(!kwargs)

		return PyResult::Ok(self);
	}

	const std::string &value() const { return m_value; }
	std::vector<int32_t> codepoints() const;
	std::optional<int32_t> codepoint() const;

	std::string to_string() const override { return m_value; }

	static PyResult __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult __repr__() const;
	PyResult __hash__() const;
	PyResult __eq__(const PyObject *obj) const;
	PyResult __lt__(const PyObject *obj) const;

	PyResult __len__() const;
	PyResult __add__(const PyObject *obj) const;

	PyResult isalpha() const;
	PyResult isalnum() const;
	PyResult isascii() const;
	PyResult isdigit() const;
	PyResult islower() const;
	PyResult isupper() const;

	PyResult capitalize() const;
	PyResult casefold() const;
	PyResult find(PyTuple *args, PyDict *kwargs) const;
	PyResult count(PyTuple *args, PyDict *kwargs) const;
	PyResult endswith(PyTuple *args, PyDict *kwargs) const;
	PyResult join(PyTuple *args, PyDict *kwargs) const;
	PyResult lower() const;
	PyResult upper() const;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

  private:
	PyString(std::string s);

	size_t get_position_from_slice(int64_t) const;
};

}// namespace py