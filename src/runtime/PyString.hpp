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
	static PyResult<PyString *> create(const std::string &value);
	static PyResult<PyString *> create(PyString *self, PyTuple *args, PyDict *kwargs)
	{
		// FIXME with proper error handling
		ASSERT(self)
		ASSERT(!args || (args->size() == 0))
		ASSERT(!kwargs)

		return Ok(self);
	}

	const std::string &value() const { return m_value; }
	std::vector<int32_t> codepoints() const;
	std::optional<int32_t> codepoint() const;

	std::string to_string() const override { return m_value; }

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<size_t> __hash__() const;
	PyResult<PyObject *> __eq__(const PyObject *obj) const;
	PyResult<PyObject *> __lt__(const PyObject *obj) const;

	PyResult<size_t> __len__() const;
	PyResult<PyObject *> __add__(const PyObject *obj) const;

	PyResult<PyObject *> isalpha() const;
	PyResult<PyObject *> isalnum() const;
	PyResult<PyObject *> isascii() const;
	PyResult<PyObject *> isdigit() const;
	PyResult<PyObject *> islower() const;
	PyResult<PyObject *> isupper() const;

	PyResult<PyObject *> capitalize() const;
	PyResult<PyObject *> casefold() const;
	PyResult<PyObject *> find(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> count(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> endswith(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> join(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> lower() const;
	PyResult<PyObject *> upper() const;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

  private:
	PyString(std::string s);

	size_t get_position_from_slice(int64_t) const;
};

}// namespace py